// Hermetic Firestore transaction/query double; optional process-restart persistence.
import {readFileSync, writeFileSync, existsSync} from 'node:fs';
const decode = (_, value) => value?.type === 'Buffer' ? Buffer.from(value.data) : value;
const copy = value => value === undefined ? undefined : JSON.parse(JSON.stringify(value), decode);
class Snapshot {
  constructor(ref, data) {this.ref = ref; this.id = ref.id; this.value = copy(data); this.exists = data !== undefined;}
  data() {return copy(this.value);}
}
class Doc {
  constructor(db, path) {this.db = db; this.path = path; this.id = path.split('/').at(-1);}
  async get() {return new Snapshot(this, this.db.values.get(this.path));}
}
class Query {
  constructor(db, path, filters = [], order = null, count = Infinity) {Object.assign(this, {db, path, filters, order, count});}
  limit(count) {return new Query(this.db, this.path, this.filters, this.order, count);}
  where(field, op, value) {return new Query(this.db, this.path, [...this.filters, {field, op, value}], this.order, this.count);}
  orderBy(field, direction) {return new Query(this.db, this.path, this.filters, {field, direction}, this.count);}
  async get() {
    let docs = [...this.db.values].filter(([path]) => path.startsWith(this.path + '/') && path.split('/').length === this.path.split('/').length + 1)
      .map(([path, value]) => new Snapshot(this.db.doc(path), value));
    for (const {field, op, value} of this.filters) docs = docs.filter(s => op === '==' ? s.data()[field] === value :
      op === 'not-in' ? s.data()[field] !== undefined && !value.includes(s.data()[field]) : false);
    if (this.order) docs = docs.filter(s => s.data()[this.order.field] !== undefined).sort((a, b) =>
      String(a.data()[this.order.field]).localeCompare(String(b.data()[this.order.field])) * (this.order.direction === 'desc' ? -1 : 1));
    return {docs: docs.slice(0, this.count)};
  }
}
export class MemoryFirestore {
  constructor(file = null) {
    this.file = file; this.values = new Map(file && existsSync(file) ? JSON.parse(readFileSync(file, 'utf8'), decode) : []);
    this.tail = Promise.resolve(); this.replay = false; this.failCommit = false;
  }
  doc(path) {return new Doc(this, path);}
  collection(path) {return new Query(this, path);}
  async runTransaction(fn) {
    const run = async () => {
      const attempt = async () => {
        const writes = [];
        const result = await fn({get: async ref => {
          if (writes.length) throw new Error('read after write');
          return ref.get();
        }, set: (ref, value, options) => writes.push({ref, value: copy(value), merge: options?.merge})});
        return {result, writes};
      };
      if (this.replay) await attempt();
      const {result, writes} = await attempt();
      if (this.failCommit) throw new Error('storage failure');
      for (const {ref, value, merge} of writes) this.values.set(ref.path, merge ? {...this.values.get(ref.path), ...value} : value);
      if (this.file) writeFileSync(this.file, JSON.stringify([...this.values]));
      return result;
    };
    const result = this.tail.then(run); this.tail = result.catch(() => {}); return result;
  }
}
