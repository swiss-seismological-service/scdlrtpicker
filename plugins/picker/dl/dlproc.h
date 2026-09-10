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


#ifndef SCDLRTPICKER_DLPROC_H
#define SCDLRTPICKER_DLPROC_H


#include "onnxmodel.h"

#include <seiscomp/processing/stream.h>
#include <seiscomp/core/record.h>
#include <seiscomp/core/datetime.h>

#include <string>
#include <type_traits>


namespace Seiscomp {
namespace Processing {


/**
 * @brief NCompsOperator PROC bridging N synchronized waveform
 *        components to an OnnxPickerStream. Used directly, with no
 *        wrapper -- unlike the classical multi-component secondary
 *        pickers (e.g. S_l2.cpp), which wrap their PROC in
 *        Operator::StreamConfigWrapper for gain correction. Samples
 *        reach this PROC, and hence the model, as raw digitizer
 *        counts.
 *
 * Unlike AIC/L2, no per-channel pre-filter is applied: the model's own
 * convolutional layers already learn whatever frequency-domain
 * features they need.
 *
 * Unlike a classical picker's PROC, this does not compute or publish
 * anything -- it only accumulates raw samples into the
 * OnnxPickerStream; inference runs once, on demand, from
 * DLPicker::process()/DLSecondaryPicker::process(), not continuously
 * here. Component 0 is still published unmodified purely so
 * NCompsOperator's store() call advances dataTimeWindow() bookkeeping,
 * which Picker::process()'s "enough data has arrived" gate depends on.
 *
 * T is constrained to double, the only sample type NCompsOperator is
 * instantiated with in this codebase.
 */
template <typename T, int N>
class DLProc {
	static_assert(std::is_same<T, double>::value,
	              "DLProc only supports T = double");

	public:
		DLProc(OnnxPickerStreamPtr stream, const Stream *streamConfig)
		: _stream(stream), _streamConfig(streamConfig) {}

		void operator()(const Record *rec, T *data[N], int n,
		                const Core::Time &stime, double sfreq) const {
			const double *channelData[N];
			for ( int i = 0; i < N; ++i ) {
				channelData[i] = data[i];
			}

			_stream->feed(channelData, n, stime, sfreq);
			// data[0] is left unchanged -- see class comment.
		}

		//! Only component 0 needs to be published/stored, purely for
		//! dataTimeWindow() bookkeeping (see class comment).
		bool publish(int c) const { return c == 0; }

		int compIndex(const std::string &code) const {
			for ( int i = 0; i < N; ++i ) {
				if ( code == _streamConfig[i].code() ) {
					return i;
				}
			}
			return -1;
		}

		//! Required by the NCompsOperator PROC concept; no
		//! translation needed, so a plain pass-through.
		const std::string &translateChannelCode(int, const std::string &code) {
			return code;
		}

		void reset() const { _stream->reset(); }

	private:
		OnnxPickerStreamPtr _stream;
		const Stream       *_streamConfig;
};


}
}


#endif
