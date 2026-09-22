import fs from 'node:fs';
import { once } from 'node:events';
import { formats, decodeMaskToRaster } from './dist/index.js';

async function writeLine(stream, value) {
  if (!stream.write(JSON.stringify(value) + '\n')) await once(stream, 'drain');
}

async function main() {
  if (process.argv.length !== 4) throw new Error('arguments_invalid');
  const input = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
  if (typeof input.text !== 'string') throw new Error('input_invalid');
  const parser = formats.segmentation.video().createParser();
  parser.push(input.text, { emit: false });
  const result = parser.finish({ status: 'completed' }).result;
  if (result.diagnostics.length || result.records.some(
    record => record.kind === 'text' && record.text.trim()
  )) throw new Error('output_malformed');

  const output = fs.createWriteStream(process.argv[3], { flags: 'wx', mode: 0o600 });
  await writeLine(output, { kind: 'begin', schema: 'sam31_rasters.v1' });
  let count = 0;
  for (const record of result.records) {
    if (record.kind !== 'mask') continue;
    const raster = decodeMaskToRaster(record.mask);
    await writeLine(output, {
      kind: 'mask', object_id: record.objectId,
      frame_index: record.frame?.frameIndex ?? -1,
      bounds: record.bounds, width: record.mask.width, height: record.mask.height,
      raster: Buffer.from(raster).toString('base64'),
    });
    count++;
  }
  await writeLine(output, { kind: 'end', mask_count: count });
  output.end();
  await once(output, 'finish');
}

main().catch(error => {
  // Never echo provider output, source pixels, or a credential on stderr.
  const code = error?.message === 'output_malformed' ? 'output_malformed' : 'failed';
  process.stderr.write(`sam31_js_parser_${code}\n`);
  process.exitCode = 1;
});
