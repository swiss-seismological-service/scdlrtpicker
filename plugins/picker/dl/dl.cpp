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


#define SEISCOMP_COMPONENT DLPicker

#include <seiscomp/logging/log.h>
#include <seiscomp/core/strings.h>
#include <seiscomp/datamodel/comment.h>
#include <seiscomp/datamodel/pick.h>
#include <seiscomp/processing/operator/ncomps.h>
#include "dlproc.h"
#include "dl.h"

#include <algorithm>
#include <cmath>


using namespace std;


namespace Seiscomp {
namespace Processing {


namespace {

// -1 is never a legitimate noiseBegin/signalBegin/signalEnd value, so
// it unambiguously means "left untouched".
const double kIgnoredWindowSentinel = -1.0;

}


// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
template <int NComponents>
DLPicker<NComponents>::DLPicker()
: _minConfidence(0.4), _lastConfidence(0.0), _strategy(Strategy::Fast),
  _polarityMinConfidence(0.5), _lastPolarityConfidence(0.0),
  _lastPolaritySet(false), _nextAttempt(0),
  _bestConfidence(-1.0), _bestPolarityConfidence(0.0),
  _bestLowerUncertainty(-1.0), _bestUpperUncertainty(-1.0),
  _bestRawConfidence(-1.0) {
	// Without this, DLPicker<3> never gets N/E data routed to it
	// (usedComponent() defaults to Vertical). Any = all available
	// components.
	setDataComponents(NComponents == 3 ? Any : Vertical);

	// Overwritten by setup(); kept as the true default if setup() were
	// ever skipped.
	setNoiseStart(kIgnoredWindowSentinel);
	setSignalStart(kIgnoredWindowSentinel);
	setSignalEnd(kIgnoredWindowSentinel);
}
// <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
template <int NComponents>
bool DLPicker<NComponents>::setup(const Settings &settings) {
	if ( !Picker::setup(settings) ) {
		return false;
	}

	string modelPath;
	try {
		modelPath = settings.getString("picker." + methodID() + ".model");
	}
	catch ( ... ) {
		SEISCOMP_ERROR("[%s] no model configured, set picker.%s.model",
		               methodID().c_str(), methodID().c_str());
		setStatus(Error, 0);
		return false;
	}

	// Process-wide cache: loads/builds the session only on the first
	// call for this path.
	OnnxSessionPtr session = OnnxSession::Load(modelPath);
	if ( !session ) {
		setStatus(Error, 0);
		return false;
	}

	if ( session->config().numComponents != NComponents ) {
		SEISCOMP_ERROR("[%s] model '%s' has %d component(s), expected %d",
		               methodID().c_str(), modelPath.c_str(),
		               session->config().numComponents, NComponents);
		setStatus(Error, 0);
		return false;
	}

	if ( std::find(session->config().phases.begin(), session->config().phases.end(), "P")
	     == session->config().phases.end() ) {
		SEISCOMP_ERROR("[%s] model '%s' does not provide a 'P' output",
		               methodID().c_str(), modelPath.c_str());
		setStatus(Error, 0);
		return false;
	}

	try { _minConfidence = settings.getDouble("picker." + methodID() + ".minConfidence"); }
	catch ( ... ) {}

	// Only meaningful if the model declares seiscomp.picker.
	// polarityLabels; see dl.h and ReadPolarity().
	try { _polarityMinConfidence = settings.getDouble("picker." + methodID() + ".polarityMinConfidence"); }
	catch ( ... ) { _polarityMinConfidence = 0.5; }

	// Confidence-derived uncertainty fallback; only used if the model
	// has no uncertaintyLabels of its own -- see dl.h and
	// ConfidenceToUncertainty(). Both must be configured together.
	_uncertaintyAtMinConfidence = Core::None;
	_uncertaintyAtMaxConfidence = Core::None;
	try { _uncertaintyAtMinConfidence = settings.getDouble("picker." + methodID() + ".uncertaintyAtMinConfidence"); }
	catch ( ... ) {}
	try { _uncertaintyAtMaxConfidence = settings.getDouble("picker." + methodID() + ".uncertaintyAtMaxConfidence"); }
	catch ( ... ) {}
	if ( _uncertaintyAtMinConfidence.has_value() != _uncertaintyAtMaxConfidence.has_value() ) {
		SEISCOMP_WARNING("[%s] uncertaintyAtMinConfidence and uncertaintyAtMaxConfidence "
		                  "must be configured together -- ignoring the one that was set",
		                  methodID().c_str());
		_uncertaintyAtMinConfidence = Core::None;
		_uncertaintyAtMaxConfidence = Core::None;
	}

	_strategy = Strategy::Fast;
	try {
		string strategy = settings.getString("picker." + methodID() + ".strategy");
		if ( strategy == "best" ) {
			_strategy = Strategy::Best;
		}
		else if ( strategy != "fast" ) {
			SEISCOMP_WARNING("[%s] picker.%s.strategy = '%s' is not recognized "
			                  "(expected 'fast' or 'best') -- using 'fast'",
			                  methodID().c_str(), methodID().c_str(), strategy.c_str());
		}
	}
	catch ( ... ) {}

	// Fresh per-instance state wrapping the shared session.
	_onnxStream = std::make_shared<OnnxPickerStream>(session);

	double windowDuration = session->windowDuration();

	// minLatency/maxLatency: seconds after the trigger where a window
	// *ends* (hence "latency"). maxLatency defaults to half the model
	// window (centred, best quality); minLatency only matters when
	// more than one window is tried.
	double minLatency = 3.0;
	double maxLatency = windowDuration * 0.5;

	try { minLatency = settings.getDouble("picker." + methodID() + ".minLatency"); }
	catch ( ... ) {}
	try { maxLatency = settings.getDouble("picker." + methodID() + ".maxLatency"); }
	catch ( ... ) {}
	if ( maxLatency < minLatency ) {
		maxLatency = minLatency;
	}

	// Auto-computed if unset so consecutive window centers are at
	// most half a model window apart (full coverage).
	int maxAttempts = 0;
	try { maxAttempts = settings.getInt("picker." + methodID() + ".maxAttempts"); }
	catch ( ... ) {}
	if ( maxAttempts < 1 ) {
		double span = maxLatency - minLatency;
		maxAttempts = span <= 0.0
		    ? 1
		    : std::min(20, (int)std::ceil(span / (windowDuration * 0.5)) + 1);
	}

	_attemptDelays.clear();
	if ( maxAttempts == 1 ) {
		// Single window, placed at maxLatency.
		_attemptDelays.push_back(maxLatency);
	}
	else {
		for ( int i = 0; i < maxAttempts; ++i ) {
			_attemptDelays.push_back(
				minLatency + (maxLatency - minLatency) * i / (maxAttempts - 1));
		}
	}
	_nextAttempt = 0;
	_bestConfidence = -1.0;
	_bestPickTime = Core::Time();
	_bestPolarity = Core::None;
	_bestPolarityConfidence = 0.0;
	_bestLowerUncertainty = -1.0;
	_bestUpperUncertainty = -1.0;
	_bestRawConfidence = -1.0;

	// noiseBegin/signalBegin/signalEnd are not used by this picker --
	// warn if configured away from the -1 sentinel.
	auto warnIfOverridden = [&](const char *key) {
		double v;
		try { v = settings.getDouble(("picker." + methodID() + "." + key).c_str()); }
		catch ( ... ) { return; }
		if ( std::fabs(v - kIgnoredWindowSentinel) > 1e-9 ) {
			SEISCOMP_WARNING("[%s] picker.%s.%s = %.3f is not used by "
			                  "deep-learning pickers and is ignored -- use "
			                  "minLatency/maxLatency/maxAttempts instead",
			                  methodID().c_str(), methodID().c_str(), key, v);
		}
	};
	warnIfOverridden("noiseBegin");
	warnIfOverridden("signalBegin");
	warnIfOverridden("signalEnd");

	// minDelay/maxDelay were renamed to minLatency/maxLatency -- warn
	// rather than silently ignore an old config.
	auto warnRenamed = [&](const char *oldKey, const char *newKey) {
		try {
			settings.getDouble(("picker." + methodID() + "." + oldKey).c_str());
			SEISCOMP_WARNING("[%s] picker.%s.%s was renamed to %s and the old "
			                  "name is ignored", methodID().c_str(),
			                  methodID().c_str(), oldKey, newKey);
		}
		catch ( ... ) {}
	};
	warnRenamed("minDelay", "minLatency");
	warnRenamed("maxDelay", "maxLatency");

	// Requested window spans every attempt: earliest start to latest end.
	setNoiseStart(_attemptDelays.front() - windowDuration);
	setSignalStart(_attemptDelays.front() - windowDuration);
	setSignalEnd(_attemptDelays.back());
	// Small safety margin only -- no filter here needs settling time.
	setMargin(Core::TimeSpan(2, 0));

	// No gain correction (unlike S_l2.cpp) -- see DLProc's class comment.
	typedef DLProc<double, NComponents> Proc;
	setOperator(new NCompsOperator<double, NComponents, Proc>(
		Proc(_onnxStream, _streamConfig)
	));

	// Resamples to the model's sampleRate unless it declared itself
	// rate-tolerant (seiscomp.picker.resample = false).
	if ( session->config().resample ) {
		_rateAdapter.reset(
			new ComponentResampler<NComponents>(session->config().sampleRate, _streamConfig));
	}

	return true;
}
// <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
template <int NComponents>
bool DLPicker<NComponents>::feed(const Record *rec) {
	if ( !_onnxStream ) {
		// Shouldn't happen before a successful setup(); fail safe
		// rather than dereference null state.
		return false;
	}

	if ( !_rateAdapter ) {
		// resample = false: pass through unresampled.
		return Picker::feed(rec);
	}

	RecordPtr resampled = _rateAdapter->feed(rec);
	if ( !resampled ) {
		// Still buffering, or an unrelated channel -- nothing to feed yet.
		return false;
	}

	// NCompsOperator itself already rejects a sampling-rate mismatch
	// across components (InvalidSamplingFreq), a safety net should
	// resampling ever be bypassed or misconfigured.
	return Picker::feed(resampled.get());
}
// <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
template <int NComponents>
const std::string &DLPicker<NComponents>::methodID() const {
	static const std::string id = NComponents == 3 ? "DL3C" : "DL1C";
	return id;
}
// <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
template <int NComponents>
const std::string &DLPicker<NComponents>::filterID() const {
	static const std::string empty;
	return empty;
}
// <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
template <int NComponents>
void DLPicker<NComponents>::process(const Record *record, const DoubleArray &) {
	if ( _stream.fsamp == 0.0 ) {
		terminate();
		return;
	}

	if ( isFinished() ) {
		return;
	}

	// For the debug logs below
	const std::string station = record->networkCode() + "." + record->stationCode();

	// Try every not-yet-attempted checkpoint whose data has arrived;
	// stop and wait for more data otherwise.
	while ( _nextAttempt < _attemptDelays.size() ) {
		Core::Time checkpoint = _trigger + Core::TimeSpan(_attemptDelays[_nextAttempt]);
		if ( dataTimeWindow().endTime() < checkpoint ) {
			return;
		}

		if ( !_onnxStream->ready(checkpoint) ) {
			// Sample count hasn't caught up yet (shouldn't normally
			// happen) -- wait.
			return;
		}

		Core::Time windowStart;
		auto probs = _onnxStream->evaluate(checkpoint, windowStart);
		++_nextAttempt;

		auto pit = probs.find("P");
		if ( pit == probs.end() || pit->second.empty() ) {
			setStatus(Error, 0.0);
			return;
		}

		const std::vector<float> &pCurve = pit->second;

		// "S" curve, if the model exports one and it's sample-aligned
		// with "P" -- used to veto P peaks that are really an S.
		auto sit = probs.find("S");
		const std::vector<float> *sCurve =
			( sit != probs.end() && sit->second.size() == pCurve.size() )
			? &sit->second : nullptr;

		for ( float v : pCurve ) {
			if ( v > _bestRawConfidence ) {
				_bestRawConfidence = v;
			}
		}

		// Pick the most confident P peak that isn't also firing as S:
		// gather local maxima clearing minConfidence, walk them by
		// confidence, take the first whose S value there stays below
		// 80% of P. Plain argmax is unsafe -- the default centred
		// window often contains the real S arrival too. The 80% bar
		// is internal, not a config knob.
		std::vector<int> candidates;
		for ( int i = 0; i < (int)pCurve.size(); ++i ) {
			if ( pCurve[i] < _minConfidence ) {
				continue;
			}
			float l = i > 0 ? pCurve[i - 1] : -1.0f;
			float r = i + 1 < (int)pCurve.size() ? pCurve[i + 1] : -1.0f;
			if ( pCurve[i] >= l && pCurve[i] >= r ) {
				candidates.push_back(i);
			}
		}
		// stable_sort keeps the earliest sample on a tie, matching
		// plain argmax.
		std::stable_sort(candidates.begin(), candidates.end(),
		                 [&pCurve](int a, int b) { return pCurve[a] > pCurve[b]; });

		int    pickIdx = -1;
		double pickConfidence = 0.0;
		for ( int i : candidates ) {
			if ( sCurve && (*sCurve)[i] >= 0.8 * pCurve[i] ) {
				SEISCOMP_DEBUG("[%s/%s] attempt %zu/%zu: P peak %.2f at sample %d "
				               "skipped -- S there (%.2f) is >= 80%% of it, "
				               "likely an S arrival",
				               methodID().c_str(), station.c_str(), _nextAttempt,
				               _attemptDelays.size(), pCurve[i], i,
				               (*sCurve)[i]);
				continue;
			}
			pickIdx = i;
			pickConfidence = pCurve[i];
			break;
		}

		if ( pickIdx < 0 ) {
			SEISCOMP_DEBUG("[%s/%s] attempt %zu/%zu: no S-free P peak reaching "
			               "minConfidence (%.2f)",
			               methodID().c_str(), station.c_str(), _nextAttempt,
			               _attemptDelays.size(), _minConfidence);
			continue;
		}

		// windowStart is evaluate()'s own boundary -- not re-derived
		// from checkpoint, to avoid disagreeing with its rounding by
		// a fraction of a sample.
		Core::Time pickTime = windowStart + Core::TimeSpan(pickIdx / _stream.fsamp);

		// Both no-ops (polarity unset, uncertainty -1) unless the
		// model declares the corresponding metadata; see dl.h.
		const auto &cfg = _onnxStream->config();
		double polarityConfidence = 0.0;
		OPT(int) polarityRow = ReadPolarity(probs, cfg.polarityLabels, pickIdx,
		                                     _polarityMinConfidence, polarityConfidence);
		OPT(Polarity) polarity;
		if ( polarityRow ) {
			polarity = *polarityRow == 0 ? POSITIVE : NEGATIVE;
		}
		double lowerUncertainty = -1.0, upperUncertainty = -1.0;
		bool hasNativeUncertainty = ReadUncertainty(probs, cfg.uncertaintyLabels, pickIdx,
		                                             lowerUncertainty, upperUncertainty);

		// Fallback for a model without uncertaintyLabels: derive a
		// symmetric uncertainty from the pick's own confidence instead.
		// See dl.h and ConfidenceToUncertainty().
		if ( lowerUncertainty < 0.0 && _uncertaintyAtMinConfidence && _uncertaintyAtMaxConfidence ) {
			lowerUncertainty = upperUncertainty = ConfidenceToUncertainty(
				pickConfidence, _minConfidence,
				*_uncertaintyAtMinConfidence, *_uncertaintyAtMaxConfidence);
		}

		SEISCOMP_DEBUG("[%s/%s] attempt %zu/%zu: confidence=%.2f polarity=%s(%.2f) "
		               "uncertainty=%.2f/%.2fs%s (ref=%s offset=%.3fs window=%.3fs)",
		               methodID().c_str(), station.c_str(), _nextAttempt, _attemptDelays.size(),
		               pickConfidence,
		               polarity ? (*polarity == POSITIVE ? "up" : "down") : "-",
		               polarityConfidence, lowerUncertainty, upperUncertainty,
		               (lowerUncertainty >= 0.0 && !hasNativeUncertainty) ? " (derived)" : "",
		               _trigger.iso().c_str(), double(windowStart - _trigger),
		               _onnxStream->windowDuration());

		if ( pickConfidence > _bestConfidence ) {
			_bestConfidence = pickConfidence;
			_bestPickTime = pickTime;
			_bestPolarity = polarity;
			_bestPolarityConfidence = polarityConfidence;
			_bestLowerUncertainty = lowerUncertainty;
			_bestUpperUncertainty = upperUncertainty;
		}

		// fast: accept the first S-free peak found. best: keep
		// evaluating; the highest-confidence one wins below.
		if ( _strategy == Strategy::Fast ) {
			acceptPick(record, pickTime, pickConfidence, polarity,
			           polarityConfidence, lowerUncertainty, upperUncertainty);
			return;
		}
	}

	// All attempts exhausted: accept the best S-free peak if it
	// cleared minConfidence, else reject with the strongest raw P seen.
	if ( _bestConfidence >= _minConfidence ) {
		acceptPick(record, _bestPickTime, _bestConfidence, _bestPolarity,
		           _bestPolarityConfidence, _bestLowerUncertainty, _bestUpperUncertainty);
	}
	else {
		_lastConfidence = _bestRawConfidence;
		setStatus(LowSNR, _bestRawConfidence);
	}
}
// <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
template <int NComponents>
void DLPicker<NComponents>::acceptPick(const Record *record, const Core::Time &pickTime,
                                        double confidence, OPT(Polarity) polarity,
                                        double polarityConfidence, double lowerUncertainty,
                                        double upperUncertainty) {
	_lastConfidence = confidence;
	_lastPolarityConfidence = polarity ? polarityConfidence : 0.0;
	_lastPolaritySet = (bool)polarity;
	setStatus(Finished, 100.);

	Result res;
	res.record = record;
	// Result has no separate "confidence" field; reusing snr to carry
	// it through to emitPick() (finalizePick() also attaches it
	// honestly as a "confidence" comment).
	res.snr = confidence;
	res.time = pickTime;
	res.timeLowerUncertainty = lowerUncertainty;
	res.timeUpperUncertainty = upperUncertainty;
	res.timeWindowBegin = double(timeWindow().startTime() - pickTime);
	res.timeWindowEnd = double(timeWindow().endTime() - pickTime);
	// Unset (as with AR-AIC/BK) unless the model declares
	// seiscomp.picker.polarityLabels; see dl.h.
	res.polarity = polarity;
	emitPick(res);
}
// <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
template <int NComponents>
bool DLPicker<NComponents>::calculatePick(
	int, const double *, int, int, int &, int &, int &, double &, OPT(Polarity) &) {
	// Unused -- process() evaluates the model directly; kept to
	// satisfy the interface.
	return false;
}
// <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
template <int NComponents>
void DLPicker<NComponents>::finalizePick(DataModel::Pick *pick) const {
	DataModel::CommentPtr comment = new DataModel::Comment();
	comment->setId("confidence");
	comment->setText(Core::toString(_lastConfidence));
	pick->add(comment.get());

	if ( _lastPolaritySet ) {
		DataModel::CommentPtr polarityComment = new DataModel::Comment();
		polarityComment->setId("polarity_confidence");
		polarityComment->setText(Core::toString(_lastPolarityConfidence));
		pick->add(polarityComment.get());
	}
}
// <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


template class DLPicker<1>;
template class DLPicker<3>;

// REGISTER_POSTPICKPROCESSOR token-pastes its argument, which must be
// a single token -- DLPicker<1> is four tokens, hence the typedefs.
typedef DLPicker<1> DLPicker1C;
typedef DLPicker<3> DLPicker3C;

REGISTER_POSTPICKPROCESSOR(DLPicker1C, "DL1C");
REGISTER_POSTPICKPROCESSOR(DLPicker3C, "DL3C");


}
}
