// One local render continuation. No provider, network, or model clients.
import fs from "node:fs";
import path from "node:path";
import crypto from "node:crypto";
import { inflateSync } from "node:zlib";

const ROLES = ["images", "target_support", "scene_without_target"];
const hash = (bytes) => `sha256:${crypto.createHash("sha256").update(bytes).digest("hex")}`;
const record = (root, file) => ({relative_path: path.relative(root, file),
  sha256: hash(fs.readFileSync(file)), size_bytes: fs.statSync(file).size});
const sorted = (v) => Array.isArray(v) ? v.map(sorted) : v && typeof v === "object"
  ? Object.fromEntries(Object.keys(v).sort().map(k => [k, sorted(v[k])])) : v;
const digest = (v, key) => {const copy = {...v}; delete copy[key]; return hash(JSON.stringify(sorted(copy)));};
function requireValue(ok, code) {if (!ok) throw new Error(`source_calibration_camera_recovery_${code}`);}
function checked(root, row) {
  requireValue(typeof row?.relative_path === "string" && !path.isAbsolute(row.relative_path), "path_invalid");
  const file = path.resolve(root, row.relative_path);
  requireValue(file.startsWith(path.resolve(root) + path.sep) && !fs.lstatSync(file).isSymbolicLink(), "path_invalid");
  const bytes = fs.readFileSync(file);
  requireValue(bytes.length === row.size_bytes && hash(bytes) === (row.sha256 ?? row.digest), "file_changed");
  return file;
}
function crc32(bytes) {
  let crc = 0xffffffff;
  for (const byte of bytes) {
    crc ^= byte;
    for (let bit=0; bit<8; bit++) crc = (crc >>> 1) ^ ((crc & 1) ? 0xedb88320 : 0);
  }
  return (crc ^ 0xffffffff) >>> 0;
}

export function decodePng(bytes, expected) {
  requireValue(bytes.length <= 64 * 1024 * 1024 && bytes.subarray(0,8).equals(Buffer.from([137,80,78,71,13,10,26,10])), "png_invalid");
  let offset=8, header=null, ended=false;
  const chunks=[];
  while (offset < bytes.length) {
    requireValue(offset + 12 <= bytes.length && !ended, "png_truncated");
    const length=bytes.readUInt32BE(offset), end=offset+12+length;
    requireValue(end <= bytes.length, "png_truncated");
    const type=bytes.toString("ascii",offset+4,offset+8), data=bytes.subarray(offset+8,end-4);
    requireValue(crc32(bytes.subarray(offset+4,end-4)) === bytes.readUInt32BE(end-4), "png_crc_invalid");
    if (type === "IHDR") {requireValue(!header && offset===8 && length===13,"png_header_invalid"); header=data;}
    else if (type === "IDAT") {requireValue(header!==null,"png_header_missing"); chunks.push(data);}
    else if (type === "IEND") {requireValue(length===0,"png_invalid"); ended=true;}
    else requireValue(type!=="tRNS" && ((type.charCodeAt(0) & 32)!==0 || type==="PLTE"), "png_unsupported_chunk");
    offset=end;
  }
  requireValue(header && ended && chunks.length, "png_incomplete");
  const width=header.readUInt32BE(0), height=header.readUInt32BE(4), channels=header[9]===2 ? 3 : header[9]===6 ? 4 : 0;
  requireValue(width===expected.width && height===expected.height && width>0 && height>0 && width*height<=4096*4096 &&
    header[8]===8 && channels && header[10]===0 && header[11]===0 && header[12]===0, "png_format_invalid");
  const stride=width*channels, raw=inflateSync(Buffer.concat(chunks), {maxOutputLength:(stride+1)*height});
  requireValue(raw.length===(stride+1)*height,"png_length_invalid");
  const pixels=Buffer.alloc(stride*height);
  for (let y=0;y<height;y++) {
    const filter=raw[y*(stride+1)]; requireValue(filter<=4,"png_filter_invalid");
    for (let x=0;x<stride;x++) {
      const left=x>=channels ? pixels[y*stride+x-channels] : 0;
      const up=y ? pixels[(y-1)*stride+x] : 0;
      const diagonal=y && x>=channels ? pixels[(y-1)*stride+x-channels] : 0;
      let predictor=0;
      if (filter===1) predictor=left;
      if (filter===2) predictor=up;
      if (filter===3) predictor=Math.floor((left+up)/2);
      if (filter===4) {
        const p=left+up-diagonal, a=Math.abs(p-left), b=Math.abs(p-up), c=Math.abs(p-diagonal);
        predictor=a<=b && a<=c ? left : b<=c ? up : diagonal;
      }
      pixels[y*stride+x]=(raw[y*(stride+1)+1+x]+predictor)&255;
    }
  }
  if (channels===3) return pixels;
  const rgb=Buffer.alloc(width*height*3);
  for (let i=0;i<width*height;i++) {
    if (channels===4) requireValue(pixels[i*4+3]===255,"png_nonopaque");
    pixels.copy(rgb,i*3,i*channels,i*channels+3);
  }
  return rgb;
}

function loadGroups(root, groups) {
  requireValue(groups.length===3 && new Set(groups.map(g=>g.role)).size===3,"groups_invalid");
  return Object.fromEntries(ROLES.map(role => {
    const group=groups.find(g=>g.role===role); requireValue(group,"role_missing");
    const file=checked(root,group.manifest), manifest=JSON.parse(fs.readFileSync(file));
    requireValue(manifest.sealed_camera_render_manifest_digest===digest(manifest,"sealed_camera_render_manifest_digest"),"manifest_changed");
    return [role,{group,manifest,root:path.dirname(file)}];
  }));
}
function measure(root, groups, cameras, gate, dimensions, round) {
  const loaded=loadGroups(root,groups);
  const rows=cameras.map(camera => {
    const files={}, images={};
    for (const role of ROLES) {
      const source=loaded[role], row=source.manifest.renders.find(r=>r.camera_id===camera.camera_id);
      requireValue(row,"frame_missing");
      const file=checked(source.root,row); files[role]=record(root,file);
      images[role]=decodePng(fs.readFileSync(file),dimensions);
    }
    let supported=0, visible=0, sum=0, squares=0;
    const rgb=images.images, support=images.target_support, background=images.scene_without_target;
    for (let i=0;i<rgb.length;i+=3) {
      const supportPixel=Math.max(support[i],support[i+1],support[i+2])>=gate.support_threshold_8bit;
      const changed=Math.max(Math.abs(rgb[i]-background[i]),Math.abs(rgb[i+1]-background[i+1]),Math.abs(rgb[i+2]-background[i+2]))>=gate.visual_contribution_threshold_8bit;
      if (supportPixel) supported++;
      if (supportPixel && changed) visible++;
      for (let c=0;c<3;c++) {sum+=rgb[i+c]; squares+=rgb[i+c]*rgb[i+c];}
    }
    const pixelStd=Math.sqrt(Math.max(0,squares/rgb.length-(sum/rgb.length)**2));
    const fraction=visible/(dimensions.width*dimensions.height);
    return {candidate_camera_id:camera.camera_id,round,frames:files,total_pixels:dimensions.width*dimensions.height,
      support_pixels:supported,support_contribution_pixels:visible,visible_fraction:fraction,
      frame_digests:Object.fromEntries(ROLES.map(role=>[role,files[role].sha256])),
      rgb_std:pixelStd,passed:fraction>=gate.minimum_visible_target_fraction && pixelStd>=1};
  });
  return {rows,loaded};
}

export function validateRecovery(recovery, runtime, originalCameras, cameraSpecs) {
  requireValue(recovery?.schema_version==="source_calibration_camera_recovery.v1" && recovery.maximum_rounds===1,"contract_invalid");
  const gate=recovery.visibility_gate;
  requireValue(gate?.support_threshold_8bit===24 && gate.visual_contribution_threshold_8bit===8 && gate.minimum_visible_target_fraction===0.01,"gate_invalid");
  const reserve=JSON.parse(fs.readFileSync(checked(runtime,recovery.replacement_camera_contract)));
  requireValue(Array.isArray(reserve),"reserve_invalid");
  const specs=reserve.length ? cameraSpecs(reserve) : [];
  requireValue(reserve.length<=16 && specs.every(c=>c.spec.intrinsics.width===1280 && c.spec.intrinsics.height===1280) &&
    reserve.every(c=>/^[a-zA-Z0-9_-]+$/.test(c.camera_id) && !originalCameras.some(o=>o.camera_id===c.camera_id)),"reserve_invalid");
  return reserve;
}

export async function resolveCameraRecovery({request, output, originalCameras, originalCameraFile, reserveCameras, originalGroups, renderReserve, cameraSpecs}) {
  const initial=measure(output,originalGroups,originalCameras,request.camera_recovery.visibility_gate,request.dimensions,0);
  let replacementGroups=[], replacement=null, rounds=0;
  if (initial.rows.some(row=>!row.passed) && reserveCameras.length) {
    rounds=1; replacementGroups=await renderReserve();
    replacement=measure(output,replacementGroups,reserveCameras,request.camera_recovery.visibility_gate,request.dimensions,1);
  }
  const available=(replacement?.rows || []).filter(row=>row.passed), selection=[];
  for (const row of initial.rows) {
    const candidate=row.passed ? row : available.shift();
    if (candidate) selection.push({camera_id:row.candidate_camera_id,candidate_camera_id:candidate.candidate_camera_id});
  }
  const resolution={schema_version:"source_calibration_camera_resolution.v1",original_render_groups:originalGroups,
    replacement_render_groups:replacementGroups,selection,measurement_rows:[...initial.rows,...(replacement?.rows || [])],
    maximum_rounds:1,rounds_used:rounds};
  if (selection.length!==16) return {status:"blocked",groups:[],camera_resolution:resolution};
  const resolved=selection.map(({camera_id,candidate_camera_id})=>({...([...originalCameras,...reserveCameras]
    .find(c=>c.camera_id===candidate_camera_id)),camera_id}));
  const cameraFile=path.join(output,"resolved_cameras.v1.json");
  if (rounds===0) fs.copyFileSync(originalCameraFile,cameraFile);
  else fs.writeFileSync(cameraFile,`${JSON.stringify(resolved)}\n`);
  resolution.resolved_camera_file=record(output,cameraFile);
  if (rounds===0) return {status:"completed",groups:originalGroups,camera_resolution:resolution};
  const groups=[];
  for (const role of ROLES) {
    const root=path.join(output,"renders","source-calibration-resolved",role), frames=path.join(root,"frames");
    fs.mkdirSync(frames,{recursive:true});
    const manifest=structuredClone(initial.loaded[role].manifest), renders=[];
    for (const {camera_id,candidate_camera_id} of selection) {
      const source=originalCameras.some(c=>c.camera_id===candidate_camera_id) ? initial.loaded[role] : replacement.loaded[role];
      const row=source.manifest.renders.find(r=>r.camera_id===candidate_camera_id);
      const destination=path.join(frames,`${camera_id}.png`);
      fs.copyFileSync(checked(source.root,row),destination);
      renders.push({...row,camera_id,source_candidate_camera_id:candidate_camera_id,relative_path:path.relative(root,destination)});
    }
    manifest.renders=renders; manifest.calibrated_cameras=cameraSpecs(resolved);
    manifest.calibrated_camera_file={digest:resolution.resolved_camera_file.sha256,binding:"resolved_frozen_candidate_camera_file",camera_count:16};
    manifest.sealed_camera_render_manifest_digest=digest(manifest,"sealed_camera_render_manifest_digest");
    const file=path.join(root,"sealed_camera_render_manifest.v1.json");
    fs.writeFileSync(file,`${JSON.stringify(manifest,null,2)}\n`);
    groups.push({role,manifest:record(output,file)});
  }
  return {status:"completed",groups,camera_resolution:resolution};
}
