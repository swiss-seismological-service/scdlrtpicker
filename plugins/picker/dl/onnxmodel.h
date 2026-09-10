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


#ifndef SCDLRTPICKER_ONNXMODEL_H
#define SCDLRTPICKER_ONNXMODEL_H


#include <seiscomp/core/datetime.h>
#include <seiscomp/core/optional.h>

#include <map>
#include <memory>
#include <string>
#include <vector>


namespace Seiscomp {
namespace Processing {


/**
 * @brief Wraps a loaded ONNX Runtime session for a windowed seismic
 *        phase-picking model (e.g. PhaseNet, EQTransformer): the
 *        model file, its ONNX graph/session, and its seiscomp.picker.*
 *        contract (see below). Holds no per-picker mutable state (see
 *        OnnxPickerStream for that), so one OnnxSession can be safely
 *        shared across every DLPicker/DLSecondaryPicker pointing at
 *        the same model file for the life of the process, rather than
 *        re-loading the file on every Detector trigger.
 *
 * Load() maintains a process-wide cache keyed by file path. Safe to
 * call runInference() concurrently from multiple threads/pickers
 * sharing the same session (ONNX Runtime documents Run() as
 * thread-safe once the session itself is no longer mutated).
 *
 * Model contract, so that any compatible model can be dropped in
 * without touching this code:
 *  - Exactly one input tensor of shape [1, numComponents, windowLength]
 *    (batch, channel, sample). Components are provided in the
 *    canonical order [vertical, first horizontal, second horizontal],
 *    resolved from the station's inventory orientation (traces not
 *    rotated). A model expecting another order declares it with
 *    seiscomp.picker.componentOrder (see Optional below).
 *  - Exactly one output tensor of shape [1, numPhases, windowLength],
 *    one row of per-sample probabilities per entry of "phases", in
 *    the same order, aligned sample-for-sample with the input.
 *  - Required ONNX metadata_props (see onnx.helper.set_model_props()
 *    at export time):
 *      seiscomp.picker.numComponents   "1" or "3"
 *      seiscomp.picker.sampleRate      input rate, e.g. "100"
 *      seiscomp.picker.windowLength    samples, e.g. "3001"
 *      seiscomp.picker.phases          comma separated, e.g. "P,S"
 *    Optional:
 *      seiscomp.picker.normalization   "std" (default) | "peak" | "none"
 *      seiscomp.picker.componentOrder  a permutation of the first
 *        numComponents letters of "ZNE" (default "ZNE" / "Z"), e.g.
 *        "ENZ" means input row 0 = second horizontal, 1 = first
 *        horizontal, 2 = vertical.
 *      seiscomp.picker.resample        "true" (default) | "false" --
 *        when false, each component is fed at its native rate
 *        unresampled (only correct if that rate is close enough to
 *        sampleRate, since sampleRate still determines
 *        windowDuration()).
 *      seiscomp.picker.polarityLabels  two names from "phases",
 *        "<up-row>,<down-row>", e.g. "Polarity_U,Polarity_D" -- rows
 *        giving first-motion polarity at the same sample index as the
 *        picked phase. P side only (DLPicker); ignored by
 *        DLSecondaryPicker, since SecondaryPicker::Result has no
 *        polarity field.
 *      seiscomp.picker.uncertaintyLabels  one or two names from
 *        "phases", e.g. "P_lower,P_upper" (or just "P_sigma" for a
 *        symmetric value) -- row(s) giving the picked phase's time
 *        uncertainty, in seconds, at the same sample index as the
 *        pick. One name means the same value is used for both lower
 *        and upper uncertainty.
 *
 * Every component reaches the model as raw digitizer counts, matching
 * the classical single-component AIC/BK/GFZ pickers -- there is no
 * gain correction anywhere in this plugin.
 *
 * DLPicker/DLSecondaryPicker request one windowLength-sized window per
 * evaluation attempt, at each of a small, bounded number of windows
 * (see DLPicker's class comment). The picker's effective search range
 * is capped at windowLength either way: a phase outside every
 * attempted window cannot be found, since the model never sees data
 * beyond its own window.
 */
class OnnxSession {
	public:
		using Ptr = std::shared_ptr<OnnxSession>;

		enum Normalization { None, Peak, StdDev };

		struct Config {
			int                      numComponents = 0;
			double                   sampleRate = 0.0;
			int                      windowLength = 0;
			std::vector<std::string> phases;
			Normalization            normalization = StdDev;
			bool                     resample = true;

			// Index of the plugin's canonical component (0 vertical,
			// 1 first horizontal, 2 second horizontal) feeding each
			// model input row. Identity unless componentOrder reorders
			// it (e.g. "ENZ" -> {2,1,0}). Size == numComponents.
			std::vector<int>         componentOrder;

			// [up, down] row names for first-motion polarity, or
			// empty if the model declares none. See ReadPolarity().
			std::vector<std::string> polarityLabels;

			// [lower] or [lower, upper] row names for the picked
			// phase's time uncertainty, or empty if the model
			// declares none. See ReadUncertainty().
			std::vector<std::string> uncertaintyLabels;
		};


	public:
		//! Returns the cached session for this path if already loaded
		//! (process-wide, thread-safe), otherwise loads and validates
		//! the file against the contract above and caches it. Returns
		//! nullptr and logs an error on failure; a failed load is not
		//! cached, so a config fix is picked up on the next call.
		static Ptr Load(const std::string &path);

		~OnnxSession();

		const Config &config() const { return _config; }

		//! Window duration in seconds (windowLength / sampleRate).
		double windowDuration() const;

		/**
		 * @brief Runs one inference over a full [numComponents,
		 *        windowLength] window (already normalized by the
		 *        caller) and returns, per phase (in config().phases
		 *        order), windowLength probability samples.
		 *
		 * @param input Flattened [numComponents, windowLength] input,
		 *              channel-major (input[c*windowLength + i]).
		 */
		std::map<std::string, std::vector<float>>
		runInference(const std::vector<float> &input) const;


	private:
		OnnxSession();

		bool load(const std::string &path);


	private:
		struct Impl;
		std::unique_ptr<Impl> _impl;

		Config _config;
};


using OnnxSessionPtr = OnnxSession::Ptr;


/**
 * @brief Per-picker-instance raw sample buffer wrapping a (possibly
 *        shared) OnnxSession, running inference exactly once per
 *        Detector trigger -- not a sliding window, since a Picker
 *        only calls in once it already has the full requested window.
 *
 * A DLPicker/DLSecondaryPicker gets a fresh OnnxPickerStream on every
 * trigger; the OnnxSession it wraps is the process-wide cached one.
 */
class OnnxPickerStream {
	public:
		explicit OnnxPickerStream(OnnxSessionPtr session);

		const OnnxSession::Config &config() const { return _session->config(); }
		double windowDuration() const { return _session->windowDuration(); }

		//! Clears all internal buffers, e.g. after a gap.
		void reset();

		//! Appends n new, already time-aligned samples (raw digitizer
		//! counts) of each configured component to this instance's
		//! private raw history. Does not run inference -- see
		//! ready()/evaluate().
		//! @param channelData Pointer to config().numComponents arrays
		//!                    of n samples each, ordered Z[,N,E].
		//! @param stime Absolute time of channelData[*][0]; tags the
		//!              start of the history buffer on the first call
		//!              after construction/reset().
		//! @param sfreq Sampling rate of this block, samples/second.
		void feed(const double * const *channelData, int n,
		          const Core::Time &stime, double sfreq);

		/**
		 * @brief True once history covers a full windowLength window
		 *        ending at (at most) "at", i.e. evaluate(at) can be
		 *        served.
		 *
		 * Time-based rather than "windowLength samples arrived
		 * somewhere": each evaluation attempt must see the window
		 * ending at its own checkpoint time, not whatever happens to
		 * be buffered when it's reached -- a shift of even a few
		 * seconds can visibly move the resulting pick.
		 */
		bool ready(const Core::Time &at) const;

		/**
		 * @brief Runs inference over the windowLength window ending
		 *        at "at" and returns windowLength probability samples
		 *        per phase (in config().phases order). Must only be
		 *        called once ready(at) is true.
		 *
		 * @param windowStart Set to the exact absolute time of the
		 *                    window's first sample. "at" is only a
		 *                    request, rounded internally to the
		 *                    nearest actual sample; callers converting
		 *                    a curve index back to an absolute time
		 *                    must use this value, not re-derive it as
		 *                    "at - windowDuration()".
		 */
		std::map<std::string, std::vector<float>> evaluate(const Core::Time &at,
		                                                    Core::Time &windowStart) const;


	private:
		OnnxSessionPtr _session;

		// Raw sample history per component, grown by feed() until
		// evaluate() consumes a windowLength-sized slice, addressed
		// by absolute time.
		std::vector<std::vector<float>> _history;

		// Absolute time of _history[*][0] and the sample rate of the
		// data in it -- set from the first feed() after construction/
		// reset(), assumed constant/contiguous afterward (a real gap
		// already triggers a full reset() upstream).
		OPT(Core::Time) _historyStart;
		double          _sampleRate = 0.0;
};


using OnnxPickerStreamPtr = std::shared_ptr<OnnxPickerStream>;


//! Reads first-motion polarity at sample idx from the two rows named
//! by polarityLabels ([up, down]). Returns unset if polarityLabels
//! isn't exactly 2 names, either row is missing from probs or too
//! short for idx, or neither value reaches minConfidence. confidence
//! is set to the winning row's value either way (0 if unset).
//! @return 0 if the "up" row won (positive first motion), 1 if "down"
//!         won (negative), unset otherwise.
OPT(int) ReadPolarity(const std::map<std::string, std::vector<float>> &probs,
                       const std::vector<std::string> &polarityLabels, int idx,
                       double minConfidence, double &confidence);

//! Reads the picked phase's time uncertainty in seconds at sample idx
//! from the row(s) named by uncertaintyLabels (one row = symmetric,
//! two = [lower, upper]). Returns false, leaving lower/upper at -1,
//! if uncertaintyLabels is empty or a named row is missing/too short.
bool ReadUncertainty(const std::map<std::string, std::vector<float>> &probs,
                      const std::vector<std::string> &uncertaintyLabels, int idx,
                      double &lower, double &upper);

//! Derives a symmetric time uncertainty (seconds) from a pick's model
//! confidence, for a model that has no seiscomp.picker.uncertaintyLabels
//! of its own. Linearly interpolates confidence in [minConfidence, 1]
//! to uncertainty in [uncertaintyAtMaxConfidence, uncertaintyAtMinConfidence],
//! clamping confidence to that range first (so a confidence at or below
//! minConfidence never happens in practice, but is handled the same as
//! minConfidence itself).
double ConfidenceToUncertainty(double confidence, double minConfidence,
                                double uncertaintyAtMinConfidence,
                                double uncertaintyAtMaxConfidence);


}
}


#endif
