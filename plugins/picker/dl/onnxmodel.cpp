/***************************************************************************
 *   Copyright (C) by ETHZ/SED                                             *
 *                                                                         *
 * This program is free software: you can redistribute it and/or modify    *
 * it under the terms of the GNU Affero General Public License as published*
 * by the Free Software Foundation, either version 3 of the License, or    *
 * (at your option) any later version.                                     *
 *                                                                         *
 * This program is distributed in the hope that it will be useful,         *
 * but WITHOUT ANY WARRANTY; without even the implied warranty of          *
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the           *
 * GNU Affero General Public License for more details.                     *
 *                                                                         *
 *                                                                         *
 *   Developed by Luca Scarabello <luca.scarabello@sed.ethz.ch>            *
 ***************************************************************************/


#define SEISCOMP_COMPONENT OnnxSession

#include <seiscomp/logging/log.h>
#include "onnxmodel.h"

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <mutex>
#include <sstream>


using namespace std;


namespace Seiscomp {
namespace Processing {


// ----------------------------------------------------------------------
//  Impl: keeps the ONNX Runtime C++ API out of onnxmodel.h so that
//  code merely using OnnxSession/OnnxPickerStream does not need
//  onnxruntime_cxx_api.h on its include path.
// ----------------------------------------------------------------------
struct OnnxSession::Impl {
	Ort::Env                     env{ORT_LOGGING_LEVEL_WARNING, "scdlrtpicker"};
	Ort::SessionOptions          sessionOptions;
	std::unique_ptr<Ort::Session> session;
	Ort::AllocatorWithDefaultOptions allocator;
	Ort::MemoryInfo              memoryInfo =
		Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

	std::string inputName;
	std::string outputName;
	std::vector<std::string> phases;

	// Runs one inference over a [1, numComponents, windowLength] input
	// and returns, per phase, a vector of windowLength probability
	// samples. Safe to call concurrently: Ort::Session::Run() is
	// thread-safe as long as the session isn't mutated after setup.
	std::map<std::string, std::vector<float>>
	run(const std::vector<float> &input, int numComponents, int windowLength) const {
		std::array<int64_t, 3> inputShape{1, numComponents, windowLength};

		// CreateTensor() takes a non-const pointer for historical
		// reasons even for inputs; it does not mutate the data.
		Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
			memoryInfo, const_cast<float *>(input.data()), input.size(),
			inputShape.data(), inputShape.size());

		const char *inNames[]  = { inputName.c_str() };
		const char *outNames[] = { outputName.c_str() };

		auto outputs = session->Run(Ort::RunOptions{nullptr},
		                             inNames, &inputTensor, 1,
		                             outNames, 1);

		const float *outData = outputs[0].GetTensorMutableData<float>();
		auto shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();

		int64_t expected = (int64_t)phases.size() * windowLength;
		int64_t actual = 1;
		for ( auto d : shape ) actual *= d;

		std::map<std::string, std::vector<float>> result;

		if ( actual != expected ) {
			SEISCOMP_ERROR("OnnxSession: unexpected output size %ld, expected %ld",
			               (long)actual, (long)expected);
			return result;
		}

		for ( size_t p = 0; p < phases.size(); ++p ) {
			result[phases[p]] = std::vector<float>(
				outData + p * windowLength, outData + (p + 1) * windowLength);
		}

		return result;
	}
};


// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
OnnxSession::OnnxSession() {}
// <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
OnnxSession::~OnnxSession() {}
// <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
OnnxSession::Ptr OnnxSession::Load(const std::string &path) {
	// Process-wide cache, keyed by path. Kept as a plain (not
	// weak_ptr) cache so a loaded session survives even between
	// infrequent triggers on a quiet station -- negligible memory
	// cost next to repeated reload cost.
	static std::mutex cacheMutex;
	static std::map<std::string, Ptr> cache;

	std::lock_guard<std::mutex> lock(cacheMutex);

	auto it = cache.find(path);
	if ( it != cache.end() ) {
		return it->second;
	}

	Ptr session(new OnnxSession());
	if ( !session->load(path) ) {
		// Not cached: a config fix followed by the next trigger
		// should retry, not keep failing on a stale nullptr.
		return nullptr;
	}

	cache[path] = session;
	return session;
}
// <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
bool OnnxSession::load(const std::string &path) {
	_impl.reset(new Impl());

	try {
		_impl->sessionOptions.SetIntraOpNumThreads(1);
		_impl->session.reset(
			new Ort::Session(_impl->env, path.c_str(), _impl->sessionOptions));
	}
	catch ( const Ort::Exception &e ) {
		SEISCOMP_ERROR("OnnxSession: failed to load '%s': %s",
		               path.c_str(), e.what());
		return false;
	}

	if ( _impl->session->GetInputCount() != 1 || _impl->session->GetOutputCount() != 1 ) {
		SEISCOMP_ERROR("OnnxSession: '%s' must have exactly one input "
		               "and one output tensor", path.c_str());
		return false;
	}

	_impl->inputName  = _impl->session->GetInputNameAllocated(0, _impl->allocator).get();
	_impl->outputName = _impl->session->GetOutputNameAllocated(0, _impl->allocator).get();

	Ort::ModelMetadata meta = _impl->session->GetModelMetadata();

	auto getString = [&](const char *key, bool required, std::string &out) -> bool {
		Ort::AllocatedStringPtr v =
			meta.LookupCustomMetadataMapAllocated(key, _impl->allocator);
		if ( !v ) {
			if ( required ) {
				SEISCOMP_ERROR("OnnxSession: '%s' missing required metadata key '%s'",
				               path.c_str(), key);
			}
			return false;
		}
		out = v.get();
		return true;
	};

	auto splitCsv = [](const std::string &s) {
		std::vector<std::string> items;
		std::stringstream ss(s);
		std::string item;
		while ( std::getline(ss, item, ',') ) {
			if ( !item.empty() ) {
				items.push_back(item);
			}
		}
		return items;
	};

	std::string tmp;

	if ( !getString("seiscomp.picker.numComponents", true, tmp) ) return false;
	_config.numComponents = std::atoi(tmp.c_str());

	if ( !getString("seiscomp.picker.sampleRate", true, tmp) ) return false;
	_config.sampleRate = std::atof(tmp.c_str());

	if ( !getString("seiscomp.picker.windowLength", true, tmp) ) return false;
	_config.windowLength = std::atoi(tmp.c_str());

	if ( !getString("seiscomp.picker.phases", true, tmp) ) return false;
	_config.phases = splitCsv(tmp);

	_config.normalization = StdDev;
	if ( getString("seiscomp.picker.normalization", false, tmp) ) {
		if ( tmp == "peak" ) _config.normalization = Peak;
		else if ( tmp == "none" ) _config.normalization = None;
		else _config.normalization = StdDev;
	}

	_config.resample = true;
	if ( getString("seiscomp.picker.resample", false, tmp) ) {
		_config.resample = !(tmp == "false" || tmp == "0");
	}

	if ( _config.numComponents != 1 && _config.numComponents != 3 ) {
		SEISCOMP_ERROR("OnnxSession: '%s' has unsupported numComponents=%d "
		               "(must be 1 or 3)", path.c_str(), _config.numComponents);
		return false;
	}

	// componentOrder: how the model's input rows map to the plugin's
	// canonical components (0 vertical, 1 first horizontal, 2 second
	// horizontal). Default identity; an explicit value is a
	// permutation of the first numComponents letters of "ZNE".
	const std::string canonicalOrder = std::string("ZNE").substr(0, _config.numComponents);
	_config.componentOrder.resize(_config.numComponents);
	for ( int i = 0; i < _config.numComponents; ++i ) {
		_config.componentOrder[i] = i;
	}
	if ( getString("seiscomp.picker.componentOrder", false, tmp) ) {
		bool ok = ((int)tmp.size() == _config.numComponents);
		std::vector<bool> used(_config.numComponents, false);
		for ( size_t r = 0; ok && r < tmp.size(); ++r ) {
			size_t slot = canonicalOrder.find(tmp[r]);
			if ( slot == std::string::npos || used[slot] ) {
				ok = false;
			}
			else {
				used[slot] = true;
				_config.componentOrder[r] = (int)slot;
			}
		}
		if ( !ok ) {
			SEISCOMP_ERROR("OnnxSession: '%s' seiscomp.picker.componentOrder "
			               "= '%s' must be a permutation of '%s'",
			               path.c_str(), tmp.c_str(), canonicalOrder.c_str());
			return false;
		}
	}

	if ( _config.windowLength <= 0 || _config.sampleRate <= 0 || _config.phases.empty() ) {
		SEISCOMP_ERROR("OnnxSession: '%s' has invalid metadata "
		               "(windowLength/sampleRate/phases)", path.c_str());
		return false;
	}

	// Both optional: named rows must actually be among "phases", or
	// ReadPolarity()/ReadUncertainty() would silently look them up in
	// an inference result that never contains them.
	auto namedRowsExist = [&](const char *key, const std::vector<std::string> &labels) {
		for ( const auto &label : labels ) {
			if ( std::find(_config.phases.begin(), _config.phases.end(), label)
			     == _config.phases.end() ) {
				SEISCOMP_ERROR("OnnxSession: '%s' %s row '%s' is not in "
				               "seiscomp.picker.phases", path.c_str(), key, label.c_str());
				return false;
			}
		}
		return true;
	};

	_config.polarityLabels.clear();
	if ( getString("seiscomp.picker.polarityLabels", false, tmp) ) {
		_config.polarityLabels = splitCsv(tmp);
		if ( _config.polarityLabels.size() != 2 ) {
			SEISCOMP_ERROR("OnnxSession: '%s' seiscomp.picker.polarityLabels "
			               "= '%s' must name exactly two rows (up,down)",
			               path.c_str(), tmp.c_str());
			return false;
		}
		if ( !namedRowsExist("seiscomp.picker.polarityLabels", _config.polarityLabels) ) {
			return false;
		}
	}

	_config.uncertaintyLabels.clear();
	if ( getString("seiscomp.picker.uncertaintyLabels", false, tmp) ) {
		_config.uncertaintyLabels = splitCsv(tmp);
		if ( _config.uncertaintyLabels.empty() || _config.uncertaintyLabels.size() > 2 ) {
			SEISCOMP_ERROR("OnnxSession: '%s' seiscomp.picker.uncertaintyLabels "
			               "= '%s' must name one or two rows", path.c_str(), tmp.c_str());
			return false;
		}
		if ( !namedRowsExist("seiscomp.picker.uncertaintyLabels", _config.uncertaintyLabels) ) {
			return false;
		}
	}

	_impl->phases = _config.phases;

	std::string orderStr;
	for ( int r : _config.componentOrder ) orderStr += canonicalOrder[r];

	SEISCOMP_INFO("OnnxSession: loaded '%s': %d component(s) [%s], %.3f sps, "
	              "window=%d (%.1fs), resample=%s, polarity=%s, uncertainty=%s",
	              path.c_str(), _config.numComponents, orderStr.c_str(), _config.sampleRate,
	              _config.windowLength, _config.windowLength / _config.sampleRate,
	              _config.resample ? "true" : "false",
	              _config.polarityLabels.empty() ? "no" : "yes",
	              _config.uncertaintyLabels.empty() ? "no" : "yes");

	return true;
}
// <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
double OnnxSession::windowDuration() const {
	return _config.windowLength / _config.sampleRate;
}
// <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
std::map<std::string, std::vector<float>>
OnnxSession::runInference(const std::vector<float> &input) const {
	return _impl->run(input, _config.numComponents, _config.windowLength);
}
// <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


// ------------------------------------------------------------------------
//  OnnxPickerStream
// ------------------------------------------------------------------------


// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
OnnxPickerStream::OnnxPickerStream(OnnxSessionPtr session)
: _session(session) {
	_history.resize(_session->config().numComponents);
}
// <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
void OnnxPickerStream::reset() {
	for ( auto &h : _history ) h.clear();
	_historyStart = Core::None;
	_sampleRate = 0.0;
}
// <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
void OnnxPickerStream::feed(const double * const *channelData, int n,
                             const Core::Time &stime, double sfreq) {
	const int C = _session->config().numComponents;

	if ( !_historyStart ) {
		_historyStart = stime;
		_sampleRate = sfreq;
	}

	for ( int c = 0; c < C; ++c ) {
		_history[c].reserve(_history[c].size() + n);
		for ( int i = 0; i < n; ++i ) {
			_history[c].push_back((float)channelData[c][i]);
		}
	}
}
// <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
bool OnnxPickerStream::ready(const Core::Time &at) const {
	if ( !_historyStart || _sampleRate <= 0.0 ) {
		return false;
	}
	const int L = _session->config().windowLength;
	// One-past-the-end index of the window ending at "at", rounded to
	// the nearest sample.
	long endIdx = std::lround((at - *_historyStart).length() * _sampleRate) + 1;
	return endIdx >= L && (size_t)endIdx <= _history[0].size();
}
// <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
std::map<std::string, std::vector<float>> OnnxPickerStream::evaluate(const Core::Time &at,
                                                                       Core::Time &windowStart) const {
	const auto &cfg = _session->config();
	const int C = cfg.numComponents;
	const int L = cfg.windowLength;

	// Same index computation as ready(at) -- must only be called once
	// ready(at) is true, which guarantees offset >= 0 and
	// offset + L <= history size.
	long endIdx = std::lround((at - *_historyStart).length() * _sampleRate) + 1;
	int offset = (int)endIdx - L;

	// The authoritative window start: exact, not a re-derivation of
	// "at - windowDuration()" that could disagree with the rounding
	// above by up to a sample.
	windowStart = *_historyStart + Core::TimeSpan(offset / _sampleRate);

	std::vector<float> input((size_t)C * L);

	for ( int c = 0; c < C; ++c ) {
		// Model input row c is fed by the plugin's canonical
		// component cfg.componentOrder[c] (identity unless
		// componentOrder reorders it).
		const std::vector<float> &comp = _history[cfg.componentOrder[c]];

		if ( cfg.normalization == OnnxSession::None ) {
			for ( int i = 0; i < L; ++i ) {
				input[(size_t)c * L + i] = (float)comp[offset + i];
			}
			continue;
		}

		// Matches SeisBench's own annotate_batch_pre(): it always
		// demeans first, unconditionally, before std/peak scaling --
		// so "peak" here means peak-of-the-demeaned-signal.
		double sum = 0.0;
		for ( int i = 0; i < L; ++i ) {
			sum += comp[offset + i];
		}
		double mean = sum / L;

		double sumsq = 0.0, maxAbs = 0.0;
		for ( int i = 0; i < L; ++i ) {
			double d = comp[offset + i] - mean;
			sumsq += d * d;
			maxAbs = std::max(maxAbs, std::fabs(d));
		}
		// Unbiased (N-1) estimator, matching PyTorch's Tensor::std().
		double sd = std::sqrt(std::max(1e-12, sumsq / std::max(1, L - 1)));
		double peak = std::max(1e-10, maxAbs);

		for ( int i = 0; i < L; ++i ) {
			double d = comp[offset + i] - mean;
			double x = d;
			switch ( cfg.normalization ) {
				case OnnxSession::StdDev: x = d / sd; break;
				case OnnxSession::Peak:   x = d / peak; break;
				default: break;
			}
			input[(size_t)c * L + i] = (float)x;
		}
	}

	return _session->runInference(input);
}
// <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
OPT(int) ReadPolarity(const std::map<std::string, std::vector<float>> &probs,
                       const std::vector<std::string> &polarityLabels, int idx,
                       double minConfidence, double &confidence) {
	confidence = 0.0;
	if ( polarityLabels.size() != 2 || idx < 0 ) {
		return Core::None;
	}

	auto up = probs.find(polarityLabels[0]);
	auto down = probs.find(polarityLabels[1]);
	if ( up == probs.end() || down == probs.end() ||
	     (size_t)idx >= up->second.size() || (size_t)idx >= down->second.size() ) {
		return Core::None;
	}

	double u = up->second[idx], d = down->second[idx];
	confidence = std::max(u, d);
	if ( confidence < minConfidence ) {
		return Core::None;
	}
	return u >= d ? 0 : 1;
}
// <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
bool ReadUncertainty(const std::map<std::string, std::vector<float>> &probs,
                      const std::vector<std::string> &uncertaintyLabels, int idx,
                      double &lower, double &upper) {
	lower = upper = -1;
	if ( uncertaintyLabels.empty() || idx < 0 ) {
		return false;
	}

	auto read = [&](const std::string &name, double &out) {
		auto it = probs.find(name);
		if ( it == probs.end() || (size_t)idx >= it->second.size() ) {
			return false;
		}
		out = it->second[idx];
		return true;
	};

	bool ok = read(uncertaintyLabels[0], lower);
	if ( ok && uncertaintyLabels.size() == 1 ) {
		upper = lower;
	}
	else if ( ok ) {
		ok = read(uncertaintyLabels[1], upper);
	}
	if ( !ok ) {
		lower = upper = -1;
	}
	return ok;
}
// <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
double ConfidenceToUncertainty(double confidence, double minConfidence,
                                double uncertaintyAtMinConfidence,
                                double uncertaintyAtMaxConfidence) {
	double span = 1.0 - minConfidence;
	double t = span > 0.0 ? (confidence - minConfidence) / span : 1.0;
	t = std::max(0.0, std::min(1.0, t));
	return uncertaintyAtMinConfidence +
	       t * (uncertaintyAtMaxConfidence - uncertaintyAtMinConfidence);
}
// <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


}
}
