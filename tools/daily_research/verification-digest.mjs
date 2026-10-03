// Portable verification binding only. Original raw packet hashes stay unchanged.
import {createHash} from 'node:crypto';
const order=(a,b)=>{const x=Array.from(a,c=>c.codePointAt(0)),y=Array.from(b,c=>c.codePointAt(0));
  for(let i=0;i<Math.min(x.length,y.length);i++)if(x[i]!==y[i])return x[i]-y[i];return x.length-y.length;};
export function verificationDigest(value) {
  const encode=x=>{
    if(typeof x==='number') {
      if(!Number.isFinite(x) || Number.isInteger(x)&&!Number.isSafeInteger(x)) throw new Error('verification_number_invalid');
      const bytes=Buffer.alloc(8);bytes.writeDoubleBE(x);return 'n:'+bytes.toString('hex');
    }
    if(Array.isArray(x))return '['+x.map(encode).join(',')+']';
    if(x&&typeof x==='object')return '{'+Object.keys(x).sort(order).map(k=>encode(k)+':'+encode(x[k])).join(',')+'}';
    const json=JSON.stringify(x);if(json===undefined)throw new Error('verification_value_invalid');
    return json.replace(/[\u007f-\uffff]/g,c=>'\\u'+c.charCodeAt(0).toString(16).padStart(4,'0'));
  };
  return createHash('sha256').update(encode(value)).digest('hex');
}
