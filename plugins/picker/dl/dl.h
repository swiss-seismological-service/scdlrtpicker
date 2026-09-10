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


#ifndef SCDLRTPICKER_DL_H
#define SCDLRTPICKER_DL_H


#include <seiscomp/processing/picker.h>
#include "onnxmodel.h"
#include "resampler.h"

#include <memory>
#include <vector>


namespace Seiscomp {
namespace Processing {


/**
 * @brief Deep-learning P picker running an arbitrary ONNX model (see
 *        OnnxSession) once scautopick's STA/LTA Detector has triggered.
 *        Registered as "DL1C" (vertical only) and "DL3C" (Z+N+E);
 *        NComponents must match the model's
 *        seiscomp.picker.numComponents, or setup() fails.
 *
 * Window placement replaces the classical noiseBegin/signalBegin/
 * signalEnd (setup() warns and ignores those if set):
 *   - minLatency/maxLatency: seconds after the trigger where the
 *     earliest/latest evaluation window ends. maxLatency defaults to
 *     half the model window (centred on the trigger, best quality).
 *   - maxAttempts: number of windows spread over [minLatency,
 *     maxLatency]; auto-computed if unset.
 *   - strategy: "fast" accepts the first window clearing
 *     minConfidence; "best" evaluates every window and keeps the most
 *     confident.
 *
 * feed() resamples each component to the model's sampleRate first
 * (skipped if seiscomp.picker.resample = false).
 *
 * If the model also exports "S", a P peak is rejected when the S curve
 * at that same sample is >= 80% of it -- guards against a window that
 * also contains the real S arrival (see process()).
 *
 * If the model also declares seiscomp.picker.polarityLabels, the
 * first-motion polarity at the picked sample is read and set on the
 * Result whenever it reaches polarityMinConfidence (default 0.5),
 * left unset otherwise (see onnxmodel.h::ReadPolarity()). If it
 * declares seiscomp.picker.uncertaintyLabels, the picked sample's
 * time uncertainty is read the same way (see ReadUncertainty());
 * without either, both are left unset/-1, same as before these were
 * supported. Neither requires any config beyond the model itself,
 * except polarityMinConfidence.
 *
 * A model without uncertaintyLabels can still get a derived uncertainty
 * by configuring uncertaintyAtMinConfidence/uncertaintyAtMaxConfidence
 * (both required together, unset by default): the pick's own confidence
 * is linearly mapped from [minConfidence, 1] to
 * [uncertaintyAtMaxConfidence, uncertaintyAtMinConfidence] seconds (see
 * onnxmodel.h::ConfidenceToUncertainty()). Ignored whenever the model
 * already provides uncertaintyLabels -- a real per-sample value always
 * wins over this heuristic.
 */
template <int NComponents>
class DLPicker : public Picker {
	public:
		enum class Strategy { Fast, Best };

		DLPicker();

	public:
		bool setup(const Settings &settings) override;

		const std::string &methodID() const override;
		const std::string &filterID() const override;

		void finalizePick(DataModel::Pick *pick) const override;

		// Resamples to the model's sampleRate before Picker::feed();
		// see resampler.h.
		bool feed(const Record *rec) override;

	protected:
		// Overridden instead of the base Picker::process(): that gates
		// acceptance on snrMin (wrong semantics for a [0,1] confidence)
		// and drives calculatePick() over continuousData(); this
		// evaluates the ONNX model directly against its own window
		// instead.
		void process(const Record *record, const DoubleArray &filteredData) override;

		// Pure virtual in Picker; unused here -- process() evaluates
		// the model directly.
		bool calculatePick(int n, const double *data,
		                   int signalStartIdx, int signalEndIdx,
		                   int &triggerIdx, int &lowerUncertainty,
		                   int &upperUncertainty, double &snr,
		                   OPT(Polarity) &polarity) override;

	private:
		// Emits a Result for an already-evaluated attempt; shared by
		// both strategy paths in process(). polarity/lowerUncertainty/
		// upperUncertainty come from ReadPolarity()/ReadUncertainty()
		// at the winning attempt's pick sample; polarity unset or
		// uncertainty -1 if the model doesn't declare them.
		void acceptPick(const Record *record, const Core::Time &pickTime,
		                 double confidence, OPT(Polarity) polarity,
		                 double polarityConfidence, double lowerUncertainty,
		                 double upperUncertainty);

	private:
		// Wraps the process-wide cached OnnxSession; fresh per trigger.
		OnnxPickerStreamPtr _onnxStream;
		double              _minConfidence;
		double              _lastConfidence;
		Strategy            _strategy;

		// Minimum ReadPolarity() confidence to set Result::polarity;
		// see the class comment.
		double              _polarityMinConfidence;
		double              _lastPolarityConfidence;

		// Whether the just-accepted pick's polarity was set
		bool                _lastPolaritySet;

		// Confidence-derived uncertainty fallback for a model without
		// uncertaintyLabels; both unset unless both are configured. See
		// the class comment and ConfidenceToUncertainty().
		OPT(double)         _uncertaintyAtMinConfidence;
		OPT(double)         _uncertaintyAtMaxConfidence;

		std::unique_ptr<ComponentResampler<NComponents>> _rateAdapter;

		// Window-end delays (s after trigger); computed once in setup().
		std::vector<double> _attemptDelays;
		std::size_t         _nextAttempt;

		// Best S-free P peak seen so far (and its polarity/uncertainty,
		// read at the same sample), what strategy=best accepts once
		// every attempt has run.
		double              _bestConfidence;
		Core::Time          _bestPickTime;
		OPT(Polarity)       _bestPolarity;
		double              _bestPolarityConfidence;
		double              _bestLowerUncertainty;
		double              _bestUpperUncertainty;

		// Highest raw P value seen across all attempts, for a
		// meaningful confidence on the reject (LowSNR) path.
		double              _bestRawConfidence;
};


}
}


#endif
