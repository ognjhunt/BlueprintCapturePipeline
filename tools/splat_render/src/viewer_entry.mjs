import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {GLTFLoader} from 'three/addons/loaders/GLTFLoader.js';
import {SplatMesh, SparkRenderer} from '@sparkjsdev/spark';

const renderer = new THREE.WebGLRenderer({antialias:true});
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(innerWidth, innerHeight);
renderer.setClearColor(0x10131a);
document.body.append(renderer.domElement);
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(55, innerWidth / innerHeight, .005, 100000);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
scene.add(new SparkRenderer({renderer}));
scene.add(new THREE.HemisphereLight(0xffffff, 0x778899, 3));
const status = document.querySelector('#status');
try {
  const splat = new SplatMesh({url:'/splat'});
  await splat.initialized;
  scene.add(splat);
  const bounds = splat.getBoundingBox(true);
  const center = bounds.getCenter(new THREE.Vector3());
  const radius = bounds.getBoundingSphere(new THREE.Sphere()).radius;
  const frameAll = () => {
    controls.target.copy(center);
    camera.up.set(0,1,0);
    camera.position.copy(center).add(new THREE.Vector3(1,.5,1).normalize().multiplyScalar(radius * 2.3));
    controls.update();
  };
  document.querySelector('#all').onclick = frameAll;
  const response = await fetch('/cameras');
  if (!response.ok) throw new Error('camera manifest unavailable');
  const views = await response.json();
  const selector = document.querySelector('#camera');
  views.forEach((view,i) => selector.add(new Option(`Captured view ${i + 1}`, String(i))));
  const setView = () => {
    const spec = views[Number(selector.value)].spec;
    camera.position.fromArray(spec.pos);
    camera.up.fromArray(spec.up || [0,1,0]);
    camera.fov = spec.fov;
    camera.updateProjectionMatrix();
    controls.target.fromArray(spec.target);
    controls.update();
  };
  selector.onchange = setView;
  if (views.length) setView(); else frameAll();
  let mesh = null;
  document.querySelector('#mode').onchange = async event => {
    const mode = event.target.value;
    try {
      if (mode !== 'splat' && !mesh) {
        status.textContent = 'Loading mesh…';
        mesh = (await new GLTFLoader().loadAsync('/mesh')).scene;
        scene.add(mesh);
      }
      splat.visible = mode === 'splat';
      if (mesh) {
        mesh.visible = mode !== 'splat';
        mesh.traverse(item => {
          if (item.isMesh) for (const material of [item.material].flat()) {
            material.wireframe = mode === 'wire'; material.side = THREE.DoubleSide;
            // A material-less GLB defaults to a fully metallic surface. This
            // vertex-colour inspection has no reflection environment.
            material.metalness = 0;
            material.roughness = 1;
            if (material.emissiveMap) {
              // Atlas supplies baked appearance. Ambient light must not light it twice.
              material.color.setRGB(0, 0, 0);
              material.emissive.setRGB(1, 1, 1);
              material.emissiveIntensity = 1;
            }
          }
        });
      }
      status.textContent = `${splat.numSplats.toLocaleString()} draft Gaussians`;
    } catch (error) { status.textContent = error.message; }
  };
  const query = new URLSearchParams(location.search);
  if (query.get('meshLabel')) document.querySelector('#mode option[value="mesh"]').textContent = query.get('meshLabel');
  if (['mesh', 'wire'].includes(query.get('mode'))) {
    const select = document.querySelector('#mode');
    select.value = query.get('mode');
    select.dispatchEvent(new Event('change'));
  }
  status.textContent = `${splat.numSplats.toLocaleString()} draft Gaussians`;
  window.BlueprintSceneInspection = {splat, get mesh(){return mesh;}, camera, controls, renderer};
  renderer.setAnimationLoop(() => {controls.update(); renderer.render(scene,camera);});
} catch (error) {status.textContent = `Unable to load scene: ${error.message}`;}
addEventListener('resize', () => {
  camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix(); renderer.setSize(innerWidth, innerHeight);
});
