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


#ifndef SCDLRTPICKER_S_DL_H
#define SCDLRTPICKER_S_DL_H


#include <seiscomp/processing/secondarypicker.h>
#include "onnxmodel.h"
#include "resampler.h"

#include <memory>
#include <vector>


namespace Seiscomp {
namespace Processing {


/**
 * @brief Deep-learning S picker running an arbitrary ONNX model (see
 *        OnnxSession), triggered by a reference P pick. Registered as
 *        "SDL1C" (vertical only) and "SDL3C" (Z+N+E) -- see DLPicker
 *        for the NComponents contract and the window/maxAttempts
 *        mechanism (here: minSP/maxSP are the smallest/largest S-P
 *        time to search, each window centred on an S at its target
 *        S-P time). An S whose true S-P time exceeds maxSP is not
 *        found, since the model never sees data beyond its own window.
 *
 * feed() resamples each component to the model's sampleRate first
 * (skipped if seiscomp.picker.resample = false).
 *
 * There is no SecondaryPicker::finalizePick() hook (unlike Picker), so
 * confidence is only carried through Result::snr. If the model
 * declares seiscomp.picker.uncertaintyLabels, the picked sample's
 * time uncertainty is read and set on the Result (see
 * onnxmodel.h::ReadUncertainty()); otherwise Result's
 * timeLowerUncertainty/timeUpperUncertainty are explicitly set to -1
 * (emitSPick()'s "not provided" sentinel) rather than left
 * default-constructed. No polarity: SecondaryPicker::Result has no
 * such field, so seiscomp.picker.polarityLabels is ignored here.
 *
 * A model without uncertaintyLabels can still get a derived uncertainty
 * by configuring uncertaintyAtMinConfidence/uncertaintyAtMaxConfidence
 * (both required together, unset by default) -- see DLPicker's class
 * comment and onnxmodel.h::ConfidenceToUncertainty(). Ignored whenever
 * the model already provides uncertaintyLabels.
 *
 * strategy (see DLPicker) controls whether process() accepts the
 * first window clearing minConfidence ("fast", default) or evaluates
 * every window and keeps the most confident ("best").
 */
template <int NComponents>
class DLSecondaryPicker : public SecondaryPicker {
	public:
		enum class Strategy { Fast, Best };

		DLSecondaryPicker();

	public:
		bool setup(const Settings &settings) override;

		const std::string &methodID() const override;
		const std::string &filterID() const override;

		// Resamples to the model's sampleRate before
		// SecondaryPicker::feed(); see resampler.h.
		bool feed(const Record *rec) override;

	protected:
		void process(const Record *rec, const DoubleArray &filteredData) override;

	private:
		// Emits a Result for an already-evaluated attempt; shared by
		// both strategy paths in process(). lowerUncertainty/
		// upperUncertainty come from ReadUncertainty() at the winning
		// attempt's pick sample; -1 if the model doesn't declare
		// uncertaintyLabels.
		void acceptPick(const Record *record, const Core::Time &pickTime,
		                 double confidence, double lowerUncertainty,
		                 double upperUncertainty);

	private:
		// Wraps the process-wide cached OnnxSession; fresh per
		// reference P pick.
		OnnxPickerStreamPtr _onnxStream;
		double              _minConfidence;
		Strategy            _strategy;

		// Confidence-derived uncertainty fallback for a model without
		// uncertaintyLabels; both unset unless both are configured.
		// See the class comment and ConfidenceToUncertainty().
		OPT(double)         _uncertaintyAtMinConfidence;
		OPT(double)         _uncertaintyAtMaxConfidence;

		std::unique_ptr<ComponentResampler<NComponents>> _rateAdapter;

		// Window-end delays (s after the P pick), one per target S-P
		// time in [minSP, maxSP]; computed once in setup().
		std::vector<double> _attemptDelays;
		std::size_t         _nextAttempt;

		// Best confidence (and its pick time/uncertainty, read at the
		// same sample) seen across all attempted checkpoints so far.
		double              _bestConfidence;
		Core::Time          _bestPickTime;
		double              _bestLowerUncertainty;
		double              _bestUpperUncertainty;
};


}
}


#endif
