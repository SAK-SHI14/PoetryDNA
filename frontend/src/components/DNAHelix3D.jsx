import { useEffect, useRef } from "react";
import * as THREE from "three";

const GOLD = 0xC9A84C;
const CRIMSON = 0x8B1A1A;
const TEXT_DIM = 0x6B6560;

export default function DNAHelix3D({ confidence = 50, className = "" }) {
  const mountRef = useRef(null);

  useEffect(() => {
    const container = mountRef.current;
    if (!container) return undefined;

    const scene = new THREE.Scene();
    const width = container.clientWidth;
    const height = container.clientHeight;
    const camera = new THREE.PerspectiveCamera(50, width / height, 0.1, 100);
    camera.position.set(0, 0, 8);

    const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(width, height);
    renderer.setClearColor(0x000000, 0);
    container.appendChild(renderer.domElement);

    const ambientLight = new THREE.AmbientLight(0xffffff, 0.4);
    scene.add(ambientLight);

    const keyLight = new THREE.PointLight(GOLD, 1.8, 20);
    keyLight.position.set(3, 4, 5);
    scene.add(keyLight);

    const rimLight = new THREE.PointLight(CRIMSON, 1.2, 18);
    rimLight.position.set(-3, -2, 4);
    scene.add(rimLight);

    const helixGroup = new THREE.Group();
    scene.add(helixGroup);

    const basePairs = 30;
    const helixRadius = 1.2;
    const helixHeight = 6;
    const turns = 2.5;

    const sphereGeom = new THREE.SphereGeometry(0.12, 12, 12);
    const connectorGeom = new THREE.CylinderGeometry(0.035, 0.035, 1, 6);

    const goldMat = new THREE.MeshStandardMaterial({
      color: GOLD, emissive: GOLD, emissiveIntensity: 0.25,
      roughness: 0.3, metalness: 0.8,
    });
    const crimsonMat = new THREE.MeshStandardMaterial({
      color: CRIMSON, emissive: CRIMSON, emissiveIntensity: 0.15,
      roughness: 0.4, metalness: 0.7,
    });
    const dimGoldMat = new THREE.MeshStandardMaterial({
      color: 0x4d4226, roughness: 0.8, metalness: 0.1,
    });
    const dimCrimsonMat = new THREE.MeshStandardMaterial({
      color: 0x3d0b0b, roughness: 0.8, metalness: 0.1,
    });
    const barMat = new THREE.MeshStandardMaterial({
      color: TEXT_DIM, roughness: 0.5, metalness: 0.3, transparent: true, opacity: 0.4,
    });
    const barDimMat = new THREE.MeshStandardMaterial({
      color: 0x2a2a2a, roughness: 0.8, metalness: 0.1, transparent: true, opacity: 0.2,
    });

    function createBackbone(x1, y1, z1, x2, y2, z2, material) {
      const start = new THREE.Vector3(x1, y1, z1);
      const end = new THREE.Vector3(x2, y2, z2);
      const mid = new THREE.Vector3().addVectors(start, end).multiplyScalar(0.5);
      const length = start.distanceTo(end);
      const boneGeom = new THREE.CylinderGeometry(0.04, 0.04, length, 6);
      const bone = new THREE.Mesh(boneGeom, material);
      bone.position.copy(mid);
      const dir = new THREE.Vector3().subVectors(end, start).normalize();
      const quat = new THREE.Quaternion();
      quat.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir);
      bone.setRotationFromQuaternion(quat);
      return bone;
    }

    for (let i = 0; i < basePairs; i++) {
      const t = i / (basePairs - 1);
      const angle = t * Math.PI * 2 * turns;
      const y = (t - 0.5) * helixHeight;
      const isActive = confidence > (i / basePairs) * 100;

      const ax = Math.cos(angle) * helixRadius;
      const az = Math.sin(angle) * helixRadius;
      const sphereA = new THREE.Mesh(sphereGeom, isActive ? goldMat : dimGoldMat);
      sphereA.position.set(ax, y, az);
      helixGroup.add(sphereA);

      const bx = Math.cos(angle + Math.PI) * helixRadius;
      const bz = Math.sin(angle + Math.PI) * helixRadius;
      const sphereB = new THREE.Mesh(sphereGeom, isActive ? crimsonMat : dimCrimsonMat);
      sphereB.position.set(bx, y, bz);
      helixGroup.add(sphereB);

      const dist = Math.sqrt((bx - ax) ** 2 + (bz - az) ** 2);
      const bar = new THREE.Mesh(connectorGeom, isActive ? barMat : barDimMat);
      bar.position.set((ax + bx) / 2, y, (az + bz) / 2);
      const direction = new THREE.Vector3(bx - ax, 0, bz - az).normalize();
      const quaternion = new THREE.Quaternion();
      quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), direction);
      bar.setRotationFromQuaternion(quaternion);
      bar.scale.y = dist;
      helixGroup.add(bar);

      if (i > 0) {
        const prevAngle = ((i - 1) / (basePairs - 1)) * Math.PI * 2 * turns;
        const prevY = ((i - 1) / (basePairs - 1) - 0.5) * helixHeight;
        const pax = Math.cos(prevAngle) * helixRadius;
        const paz = Math.sin(prevAngle) * helixRadius;
        helixGroup.add(createBackbone(pax, prevY, paz, ax, y, az, isActive ? goldMat : dimGoldMat));
        const pbx = Math.cos(prevAngle + Math.PI) * helixRadius;
        const pbz = Math.sin(prevAngle + Math.PI) * helixRadius;
        helixGroup.add(createBackbone(pbx, prevY, pbz, bx, y, bz, isActive ? crimsonMat : dimCrimsonMat));
      }
    }

    let isDragging = false;
    let prevMouse = { x: 0, y: 0 };
    let rotationVelocity = { x: 0, y: 0 };

    const onPointerDown = (e) => {
      isDragging = true;
      prevMouse = { x: e.clientX, y: e.clientY };
      rotationVelocity = { x: 0, y: 0 };
      renderer.domElement.style.cursor = "grabbing";
    };

    const onPointerMove = (e) => {
      if (!isDragging) return;
      const dx = e.clientX - prevMouse.x;
      const dy = e.clientY - prevMouse.y;
      rotationVelocity.x = dy * 0.005;
      rotationVelocity.y = dx * 0.005;
      helixGroup.rotation.x += rotationVelocity.x;
      helixGroup.rotation.y += rotationVelocity.y;
      prevMouse = { x: e.clientX, y: e.clientY };
    };

    const onPointerUp = () => {
      isDragging = false;
      renderer.domElement.style.cursor = "grab";
    };

    renderer.domElement.style.cursor = "grab";
    renderer.domElement.addEventListener("pointerdown", onPointerDown);
    window.addEventListener("pointermove", onPointerMove);
    window.addEventListener("pointerup", onPointerUp);

    const handleResize = () => {
      const w = container.clientWidth;
      const h = container.clientHeight;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    };
    window.addEventListener("resize", handleResize);

    let frameId = 0;
    const animate = () => {
      if (!isDragging) {
        helixGroup.rotation.y += 0.006;
        rotationVelocity.x *= 0.95;
        rotationVelocity.y *= 0.95;
        helixGroup.rotation.x += rotationVelocity.x;
        helixGroup.rotation.y += rotationVelocity.y;
      }
      renderer.render(scene, camera);
      frameId = requestAnimationFrame(animate);
    };
    animate();

    return () => {
      cancelAnimationFrame(frameId);
      window.removeEventListener("resize", handleResize);
      window.removeEventListener("pointermove", onPointerMove);
      window.removeEventListener("pointerup", onPointerUp);
      renderer.domElement.removeEventListener("pointerdown", onPointerDown);
      renderer.dispose();
      container.removeChild(renderer.domElement);
    };
  }, [confidence]);

  return (
    <div
      ref={mountRef}
      className={`h-[320px] w-full select-none ${className}`}
      aria-label="Interactive 3D DNA helix visualization"
    />
  );
}
