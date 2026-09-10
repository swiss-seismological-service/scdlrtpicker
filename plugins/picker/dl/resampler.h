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


#ifndef SCDLRTPICKER_RESAMPLER_H
#define SCDLRTPICKER_RESAMPLER_H


#include <seiscomp/io/recordfilter/resample.h>
#include <seiscomp/processing/stream.h>
#include <seiscomp/core/record.h>

#include <array>
#include <memory>
#include <string>


namespace Seiscomp {
namespace Processing {


/**
 * @brief Resamples each of N configured components (matched by
 *        channel code, same convention as DLProc::compIndex()) to a
 *        fixed target rate before records reach the picker's own
 *        WaveformProcessor::feed()/NCompsOperator pipeline, which
 *        requires all N components to already share one sampling rate.
 *
 * Handles whatever native rate a station streams at (100 Hz, 50 Hz,
 * ...), converting to the ONNX model's required
 * seiscomp.picker.sampleRate, using SeisComP's own
 * IO::RecordResampler (Lanczos-kernel FIR, already used elsewhere in
 * the framework) rather than pulling in a new dependency.
 *
 * One IO::RecordResampler<double> per component, since that class
 * does not demultiplex. An already-matching rate is a cheap
 * passthrough inside RecordResampler itself. Gaps are handled by
 * RecordResampler's own internal per-stream gap detection.
 *
 * Known limitation: IO::RecordResampler::flush() is a permanent
 * no-op, so up to one decimation period's worth of the very latest
 * samples can remain trapped in its ring buffer and never reach a
 * picker's last evaluation attempt -- a small fraction-of-a-second
 * edge effect for a windowLength of many seconds, not a correctness
 * issue.
 */
template <int N>
class ComponentResampler {
	public:
		ComponentResampler(double targetRate, const Stream *streamConfig)
		: _streamConfig(streamConfig) {
			for ( int i = 0; i < N; ++i ) {
				_resamplers[i].reset(new IO::RecordResampler<double>(targetRate));
			}
		}

		/**
		 * @brief Feeds one record and returns the resampled record if
		 *        one became available (e.g. a downsampling decimation
		 *        window filled), or nullptr otherwise (still
		 *        buffering, or the record's channel code doesn't match
		 *        any configured component).
		 *
		 * The returned record is newly allocated by RecordResampler;
		 * capturing it directly in a RecordPtr (as done here) is the
		 * correct ownership handoff -- pass RecordPtr::get() straight
		 * to WaveformProcessor::feed(), never delete it manually.
		 */
		RecordPtr feed(const Record *rec) {
			int idx = compIndex(rec->channelCode());
			if ( idx < 0 ) {
				return nullptr;
			}
			return RecordPtr(_resamplers[idx]->feed(rec));
		}

	private:
		int compIndex(const std::string &code) const {
			for ( int i = 0; i < N; ++i ) {
				if ( code == _streamConfig[i].code() ) {
					return i;
				}
			}
			return -1;
		}

	private:
		const Stream *_streamConfig;
		std::array<std::unique_ptr<IO::RecordResampler<double>>, N> _resamplers;
};


}
}


#endif
