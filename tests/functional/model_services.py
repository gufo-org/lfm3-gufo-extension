#!/usr/bin/env python3
"""Run bounded LFM3-Audio ASR/TTS HTTP checks against an owned production server.

No downloads. Each invocation creates a fresh evidence directory and retains
request bodies, model identity, configuration, timings and server logs.
"""

import argparse
from pathlib import Path
import socket
import subprocess
import time
import traceback

from run import provenance, server
from service_http import Evidence, sha256


HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('service', choices=('asr', 'tts'))
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True, help='LFM3-Audio backbone GGUF')
    parser.add_argument('--mmproj', type=Path, required=True, help='LFM3-Audio audio towers GGUF')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--audio-root', type=Path, default=HERE / 'fixtures/audio')
    parser.add_argument('--voice', choices=('female', 'male'), default='female')
    parser.add_argument('--asr-binary', type=Path, help='Explicit pinned evaluator for TTS intelligibility')
    parser.add_argument('--asr-model', type=Path)
    parser.add_argument('--startup-timeout', type=float, default=300)
    args = parser.parse_args()
    if args.service == 'tts' and not (args.asr_binary and args.asr_model):
        parser.error('TTS requires --asr-binary and --asr-model for intelligibility checks')
    if args.startup_timeout <= 0:
        parser.error('--startup-timeout must be positive')
    args.binary, args.model, args.audio_root = (path.resolve() for path in
                                               (args.binary, args.model, args.audio_root))
    if not (args.binary.is_file() and args.model.is_file() and args.mmproj.is_file()):
        parser.error('binary, --model and --mmproj must exist; missing weights are not a pass')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    with socket.socket() as reserve:
        reserve.bind(('127.0.0.1', 0))
        port = reserve.getsockname()[1]
    public_model = 'functional-' + args.service
    command = [str(args.binary), 'serve', args.service, '--model', str(args.model),
               '--host', '127.0.0.1', '--port', str(port), '--served-model-name', public_model]
    command += ['--mmproj', str(args.mmproj.resolve())]
    voice = args.voice
    identity = {path.name: sha256(path) for path in (args.model, args.mmproj.resolve())}
    weights = [args.model, args.mmproj.resolve()]
    report = {**provenance(), 'schema': 1, 'service': args.service, 'status': 'running',
              'command': command, 'model_configuration_sha256': identity,
              'source_revision': subprocess.check_output(
                  ['git', '-C', str(HERE.parents[1]), 'rev-parse', 'HEAD'], text=True).strip(),
              'weights': [{'path': str(path), 'bytes': path.stat().st_size,
                           'mtime_ns': path.stat().st_mtime_ns} for path in weights],
              'binary_sha256': sha256(args.binary), 'voice': voice if args.service == 'tts' else None,
              'timing_qualification': 'observations only; no matched baseline comparison',
              'service_harness_sha256': {name: sha256(HERE / name) for name in
                  ('model_services.py', 'service_http.py', 'asr_service.py', 'tts_service.py')}}
    evidence = Evidence(output, report)
    evidence.save()
    url = f'http://127.0.0.1:{port}'
    try:
        started = time.perf_counter()
        with server(command, output / 'server.log', args.startup_timeout) as process:
            report['startup_ms'] = (time.perf_counter() - started) * 1000
            with evidence.case('readiness-and-model'):
                evidence.json(url + '/ready', label='ready', method='GET')
                models = evidence.json(url + '/v1/models', label='models', method='GET')
                assert public_model in [item['id'] for item in models['data']], models
            if args.service == 'asr':
                import asr_service
                asr_service.run(evidence, url, args.audio_root, public_model,
                                silence_empty=False)
            else:
                import tts_service
                tts_service.run(evidence, url, public_model, voice, 'custom', args.audio_root,
                                args.asr_binary.resolve(), args.asr_model.resolve())
            assert process.poll() is None, 'server exited unexpectedly during checks'
        if args.service in ('asr', 'tts'):
            with evidence.case('server-observed-cancellation'):
                request = next(row for row in report['requests'] if row['label'] == 'cancel')
                request_id = request['headers']['x-request-id']
                lines = (output / 'server.log').read_text().splitlines()
                assert any(f'request={request_id} ' in line and 'event=completed ' in line
                           and 'error_code=cancelled' in line for line in lines), (
                               'disconnect did not cancel actual server inference', request_id)
        report['status'] = 'passed'
    except BaseException:
        report.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        evidence.save()
    print('PASS', args.service, output, flush=True)


if __name__ == '__main__':
    main()
