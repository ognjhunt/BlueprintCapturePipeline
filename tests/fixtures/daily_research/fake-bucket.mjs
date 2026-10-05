// Hermetic object-store double with object generations; optional process-restart persistence.
import {readFileSync, writeFileSync, existsSync} from 'node:fs';
export class FakeBucket {
  constructor(path = null, name = 'blueprint-8c1ca.appspot.com') {
    const saved = path && existsSync(path) ? JSON.parse(readFileSync(path, 'utf8')) : {objects: [], generation: 1000};
    this.name = name; this.path = path; this.generation = saved.generation; this.hang = false; this.failure = null;
    this.objects = new Map(saved.objects.map(([key, value]) => [key, {...value, raw: Buffer.from(value.raw, 'base64')}]));
  }
  persist() {
    if (this.path) writeFileSync(this.path, JSON.stringify({generation: this.generation,
      objects: [...this.objects].map(([key, value]) => [key, {...value, raw: value.raw.toString('base64')}])}));
  }
  async gate() {
    if (this.hang) await new Promise(() => {});
    if (this.failure) {const error = new Error('synthetic object store failure'); error.code = this.failure; throw error;}
  }
  file(name, options = undefined) {
    const bucket = this, wanted = options?.generation === undefined ? null : String(options.generation);
    const current = () => {
      const object = bucket.objects.get(name);
      if (!object || wanted !== null && String(object.generation) !== wanted) {
        const error = new Error('No such object'); error.code = 404; throw error;
      }
      return object;
    };
    return {name, async save(raw, opts) {
      await bucket.gate();
      if (opts?.preconditionOpts?.ifGenerationMatch !== 0) throw new Error('create-only precondition required');
      if (bucket.objects.has(name)) {const error = new Error('exists'); error.code = 412; throw error;}
      bucket.objects.set(name, {raw: Buffer.from(raw), generation: ++bucket.generation, metadata: opts.metadata || null});
      bucket.persist();
    }, async getMetadata() {
      await bucket.gate(); const object = current();
      return [{generation: String(object.generation), size: String(object.size ?? object.raw.length)}];
    }, async download() {
      await bucket.gate(); return [Buffer.from(current().raw)];
    }};
  }
}
