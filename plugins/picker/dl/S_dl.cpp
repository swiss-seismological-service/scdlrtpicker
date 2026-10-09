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


#define SEISCOMP_COMPONENT DLSecondaryPicker

#include <seiscomp/logging/log.h>
#include <seiscomp/processing/operator/ncomps.h>
#include "dlproc.h"
#include "S_dl.h"

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
DLSecondaryPicker<NComponents>::DLSecondaryPicker()
: _minConfidence(0.3), _strategy(Strategy::Fast), _phaseHint("S"), _nextAttempt(0), _bestConfidence(-1.0),
  _bestLowerUncertainty(-1.0), _bestUpperUncertainty(-1.0) {
	// See DLPicker's constructor: without this, DLSecondaryPicker<3>'s
	// N/E data never gets routed to it.
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
bool DLSecondaryPicker<NComponents>::setup(const Settings &settings) {
	if ( !SecondaryPicker::setup(settings) ) {
		return false;
	}

	string modelPath;
	try {
		modelPath = settings.getString("spicker." + methodID() + ".model");
	}
	catch ( ... ) {
		SEISCOMP_ERROR("[%s] no model configured, set spicker.%s.model",
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

	if ( std::find(session->config().phases.begin(), session->config().phases.end(), "S")
	     == session->config().phases.end() ) {
		SEISCOMP_ERROR("[%s] model '%s' does not provide an 'S' output",
		               methodID().c_str(), modelPath.c_str());
		setStatus(Error, 0);
		return false;
	}

	try { _minConfidence = settings.getDouble("spicker." + methodID() + ".minConfidence"); }
	catch ( ... ) {}

	// The S counterpart of scautopick's phaseHint, which only names the
	// primary (P) picker's picks.
	_phaseHint = "S";
	try { _phaseHint = settings.getString("spicker." + methodID() + ".phaseHint"); }
	catch ( ... ) {}
	if ( _phaseHint.empty() ) {
		_phaseHint = "S";
	}

	// Confidence-derived uncertainty fallback; only used if the model
	// has no uncertaintyLabels of its own -- see S_dl.h and
	// ConfidenceToUncertainty(). Unset or blank: no fallback.
	string uncertaintyMap;
	try { uncertaintyMap = settings.getString("spicker." + methodID() + ".uncertaintyMap"); }
	catch ( ... ) {}
	string uncertaintyMapError;
	if ( !ParseUncertaintyMap(uncertaintyMap, _uncertaintyMap, uncertaintyMapError) ) {
		SEISCOMP_ERROR("[%s] spicker.%s.uncertaintyMap = '%s': %s",
		               methodID().c_str(), methodID().c_str(), uncertaintyMap.c_str(),
		               uncertaintyMapError.c_str());
		setStatus(Error, 0);
		return false;
	}

	_strategy = Strategy::Fast;
	try {
		string strategy = settings.getString("spicker." + methodID() + ".strategy");
		if ( strategy == "best" ) {
			_strategy = Strategy::Best;
		}
		else if ( strategy != "fast" ) {
			SEISCOMP_WARNING("[%s] spicker.%s.strategy = '%s' is not recognized "
			                  "(expected 'fast' or 'best') -- using 'fast'",
			                  methodID().c_str(), methodID().c_str(), strategy.c_str());
		}
	}
	catch ( ... ) {}

	// Fresh per-instance state wrapping the shared session.
	_onnxStream = std::make_shared<OnnxPickerStream>(session);

	double windowDuration = session->windowDuration();

	// minLatency/maxLatency: seconds after the reference P pick where
	// the earliest/latest window *ends*, exactly as for DLPicker. Which
	// S-P times a window can pick depends on where in its window the
	// model can place an onset (see README): a model picking a centred
	// S needs latency ~ S-P + windowDuration/2, one trained on onsets
	// near the window's right edge only latency ~ S-P + a second or
	// two. The defaults sweep S-P in [0, 60] s for a centred model,
	// which suits a regional network. minLatency's default is capped
	// at maxLatency, so configuring a maxLatency below
	// windowDuration/2 alone is honoured rather than clamped back up
	// to minLatency below.
	double maxLatency = 60.0 + windowDuration * 0.5;
	try { maxLatency = settings.getDouble("spicker." + methodID() + ".maxLatency"); }
	catch ( ... ) {}

	double minLatency = std::min(windowDuration * 0.5, maxLatency);
	try { minLatency = settings.getDouble("spicker." + methodID() + ".minLatency"); }
	catch ( ... ) {}
	if ( maxLatency < minLatency ) {
		maxLatency = minLatency;
	}

	// Auto-computed if unset so consecutive window ends are at most
	// half a model window apart (full coverage).
	int maxAttempts = 0;
	try { maxAttempts = settings.getInt("spicker." + methodID() + ".maxAttempts"); }
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
	_bestLowerUncertainty = -1.0;
	_bestUpperUncertainty = -1.0;

	// noiseBegin/signalBegin/signalEnd are not used by this picker --
	// warn if configured away from the -1 sentinel.
	auto warnIfOverridden = [&](const char *key) {
		double v;
		try { v = settings.getDouble(("spicker." + methodID() + "." + key).c_str()); }
		catch ( ... ) { return; }
		if ( std::fabs(v - kIgnoredWindowSentinel) > 1e-9 ) {
			SEISCOMP_WARNING("[%s] spicker.%s.%s = %.3f is not used by "
			                  "deep-learning pickers and is ignored -- use "
			                  "minLatency/maxLatency/maxAttempts instead",
			                  methodID().c_str(), methodID().c_str(), key, v);
		}
	};
	warnIfOverridden("noiseBegin");
	warnIfOverridden("signalBegin");
	warnIfOverridden("signalEnd");

	// Requested window spans every attempt: earliest start to latest
	// end, relative to the reference P pick.
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
void DLSecondaryPicker<NComponents>::computeTimeWindow() {
	SecondaryPicker::computeTimeWindow();
	// TimeWindowProcessor::store() terminates the picker the moment a
	// record starts past _safetyTimeWindow.endTime(), so if that end time
	// is smaller than _attemptDelays.back() the later attempts never run.
	if ( !_trigger.onset.valid() || _attemptDelays.empty() ) return;
	Core::Time requiredEnd = _trigger.onset + Core::TimeSpan(_attemptDelays.back());
	if ( timeWindow().endTime() < requiredEnd ) {
		SEISCOMP_WARNING("[%s] signalEnd override (%.4fs after trigger) is shorter "
		                 "than the last attempt checkpoint (%.4fs); ignoring override",
		                 methodID().c_str(),
		                 double(timeWindow().endTime() - _trigger.onset),
		                 _attemptDelays.back());
		Core::Time startTime = timeWindow().startTime();
		setTimeWindow(Core::TimeWindow(startTime, requiredEnd));
	}
}
// <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
template <int NComponents>
bool DLSecondaryPicker<NComponents>::feed(const Record *rec) {
	if ( !_onnxStream ) {
		// Shouldn't happen before a successful setup(); fail safe
		// rather than dereference null state.
		return false;
	}

	if ( !_rateAdapter ) {
		// resample = false: pass through unresampled.
		return SecondaryPicker::feed(rec);
	}

	RecordPtr resampled = _rateAdapter->feed(rec);
	if ( !resampled ) {
		return false;
	}

	// NCompsOperator itself already rejects a sampling-rate mismatch
	// across components (InvalidSamplingFreq), a safety net should
	// resampling ever be bypassed or misconfigured.
	return SecondaryPicker::feed(resampled.get());
}
// <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
template <int NComponents>
const std::string &DLSecondaryPicker<NComponents>::methodID() const {
	static const std::string id = NComponents == 3 ? "SDL3C" : "SDL1C";
	return id;
}
// <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
template <int NComponents>
const std::string &DLSecondaryPicker<NComponents>::filterID() const {
	static const std::string empty;
	return empty;
}
// <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
template <int NComponents>
void DLSecondaryPicker<NComponents>::process(const Record *rec, const DoubleArray &) {
	// SecondaryPicker has no finished-guard of its own, but
	// WaveformProcessor::store() stops feeding data once setStatus()
	// below moves status past InProgress, so a checkpoint is never
	// re-attempted after acceptance/final rejection. See
	// DLPicker::process() for the attempt-loop rationale.

	// For the debug log below
	const std::string station = rec->networkCode() + "." + rec->stationCode();

	while ( _nextAttempt < _attemptDelays.size() ) {
		Core::Time checkpoint = _trigger.onset + Core::TimeSpan(_attemptDelays[_nextAttempt]);
		if ( dataTimeWindow().endTime() < checkpoint ) {
			return;
		}

		if ( !_onnxStream->ready(checkpoint) ) {
			return;
		}

		Core::Time windowStart;
		auto probs = _onnxStream->evaluate(checkpoint, windowStart);
		++_nextAttempt;

		auto it = probs.find("S");
		if ( it == probs.end() || it->second.empty() ) {
			setStatus(Error, 0.0);
			return;
		}

		// Only samples strictly after the reference P can be an S. An
		// early window (e.g. ending a second after P) is mostly pre-P
		// data, where a plain argmax could land on noise or on the P.
		const std::vector<float> &curve = it->second;
		int first = std::max(0, (int)std::floor(double(_trigger.onset - windowStart)
		                                        * _stream.fsamp) + 1);
		if ( first >= (int)curve.size() ) {
			SEISCOMP_DEBUG("[%s/%s] attempt %zu/%zu: window ends before the P pick, "
			               "nothing to search", methodID().c_str(), station.c_str(),
			               _nextAttempt, _attemptDelays.size());
			continue;
		}

		int best = first;
		float bestVal = curve[first];
		for ( size_t i = first + 1; i < curve.size(); ++i ) {
			if ( curve[i] > bestVal ) {
				bestVal = curve[i];
				best = (int)i;
			}
		}

		// windowStart is evaluate()'s own boundary -- not re-derived
		// from checkpoint, to avoid disagreeing with its rounding by
		// a fraction of a sample.
		Core::Time pickTime = windowStart + Core::TimeSpan(best / _stream.fsamp);

		// No-op (both -1) unless the model declares
		// seiscomp.picker.uncertaintyLabels; see S_dl.h.
		double lowerUncertainty = -1.0, upperUncertainty = -1.0;
		bool hasNativeUncertainty = ReadUncertainty(probs, _onnxStream->config().uncertaintyLabels,
		                                             best, lowerUncertainty, upperUncertainty);

		// Fallback for a model without uncertaintyLabels: derive a
		// symmetric uncertainty from the pick's own confidence instead.
		// See S_dl.h and ConfidenceToUncertainty().
		if ( lowerUncertainty < 0.0 && !_uncertaintyMap.empty() ) {
			lowerUncertainty = upperUncertainty =
				ConfidenceToUncertainty(bestVal, _uncertaintyMap);
		}

		SEISCOMP_DEBUG("[%s/%s] attempt %zu/%zu: confidence=%.2f "
		               "uncertainty=%.2f/%.2fs%s (ref=%s offset=%.3fs window=%.3fs)",
		               methodID().c_str(), station.c_str(), _nextAttempt, _attemptDelays.size(),
		               bestVal, lowerUncertainty, upperUncertainty,
		               (lowerUncertainty >= 0.0 && !hasNativeUncertainty) ? " (derived)" : "",
		               _trigger.onset.iso().c_str(), double(windowStart - _trigger.onset),
		               _onnxStream->windowDuration());

		if ( bestVal > _bestConfidence ) {
			_bestConfidence = bestVal;
			_bestPickTime = pickTime;
			_bestLowerUncertainty = lowerUncertainty;
			_bestUpperUncertainty = upperUncertainty;
		}

		if ( _strategy == Strategy::Fast && bestVal >= _minConfidence ) {
			acceptPick(rec, pickTime, bestVal, lowerUncertainty, upperUncertainty);
			return;
		}

		// Not confident enough yet (fast), or best always keeps going
		// regardless of confidence: loop continues to the next attempt.
	}

	// All attempts exhausted: accept the best attempt if it cleared
	// minConfidence, otherwise reject using its confidence.
	if ( _bestConfidence >= _minConfidence ) {
		acceptPick(rec, _bestPickTime, _bestConfidence,
		           _bestLowerUncertainty, _bestUpperUncertainty);
	}
	else {
		setStatus(LowSNR, _bestConfidence);
	}
}
// <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
template <int NComponents>
void DLSecondaryPicker<NComponents>::acceptPick(const Record *record, const Core::Time &pickTime,
                                                 double confidence, double lowerUncertainty,
                                                 double upperUncertainty) {
	Result result;
	result.time = pickTime;
	result.snr = confidence;
	result.phaseCode = _phaseHint;
	result.filterID = filterID();
	result.record = record;
	// timeLowerUncertainty/timeUpperUncertainty have no default member
	// initializer; leaving them default-constructed would hand
	// emitSPick() two uninitialized values. -1 (emitSPick()'s "not
	// provided" sentinel) unless the model declares
	// seiscomp.picker.uncertaintyLabels; see S_dl.h.
	result.timeLowerUncertainty = lowerUncertainty;
	result.timeUpperUncertainty = upperUncertainty;

	setStatus(Finished, 100);
	emitPick(result);
}
// <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


template class DLSecondaryPicker<1>;
template class DLSecondaryPicker<3>;

// REGISTER_SECONDARYPICKPROCESSOR token-pastes its argument, which
// must be a single token -- hence the typedefs.
typedef DLSecondaryPicker<1> DLSecondaryPicker1C;
typedef DLSecondaryPicker<3> DLSecondaryPicker3C;

REGISTER_SECONDARYPICKPROCESSOR(DLSecondaryPicker1C, "SDL1C");
REGISTER_SECONDARYPICKPROCESSOR(DLSecondaryPicker3C, "SDL3C");


}
}
