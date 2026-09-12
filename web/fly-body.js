// The same compiled NeuroMechFly meshes and poses in both viewers.
import * as THREE from 'three';
export function flyObjects(data) {
  const meshes=data.meshes.map(m=>{
    const g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.Float32BufferAttribute(m.vertices,3));g.setIndex(m.faces);g.computeVertexNormals();return g;
  });
  const matrix=new THREE.Matrix4();
  return data.geoms.map((g,i)=>{
    const geometry=g.type===7?meshes[g.mesh]:g.type===6?new THREE.BoxGeometry(...g.size.map(v=>v*2)):new THREE.SphereGeometry(g.size[0],16,10);
    const color=new THREE.Color().setRGB(...g.color.slice(0,3)),alpha=g.color[3];
    const o=new THREE.Mesh(geometry,new THREE.MeshStandardMaterial({color,roughness:.65,metalness:.05,opacity:alpha,transparent:alpha<1,side:THREE.DoubleSide}));
    o.name=g.name;o.position.fromArray(data.positions[i]);const r=data.rotations[i];
    matrix.set(r[0],r[1],r[2],0,r[3],r[4],r[5],0,r[6],r[7],r[8],0,0,0,0,1);o.quaternion.setFromRotationMatrix(matrix);
    if(g.type===7)detailFly(o,i);
    return o;
  });
}

// Procedural surface detail follows each original body segment in both studios.
// These display materials and bristles do not change the physical or neural models.
function detailFly(object,seed) {
  const name=object.name,eye=name.endsWith('_eye'),wing=name.endsWith('_wing');
  const material=object.material;
  if(eye){material.color.set('#942c18');material.roughness=.3;material.metalness=.12;material.flatShading=true;}
  else if(wing){material.color.set('#bfc9bd');material.opacity=.25;material.roughness=.25;material.depthWrite=false;}
  else{material.color.set(name.includes('abdomen')?'#684126':name.includes('thorax')?'#977045':name.includes('head')?'#866039':'#644522');material.roughness=.58;material.metalness=.07;}
  let state=seed+91;const random=()=>{state=(Math.imul(state,1664525)+1013904223)>>>0;return state/4294967296;};
  const positions=object.geometry.attributes.position,normals=object.geometry.attributes.normal;
  const shades=new Float32Array(positions.count*3);
  for(let i=0;i<positions.count;i++){const v=(eye ? .65 : .82)+random()*(eye ? .35 : .18);shades.set([v,v,v],i*3);}
  object.geometry.setAttribute('color',new THREE.BufferAttribute(shades,3));material.vertexColors=true;
  if(wing){const veins=new THREE.Mesh(object.geometry,new THREE.MeshBasicMaterial({color:'#64664e',wireframe:true,transparent:true,opacity:.055,depthWrite:false}));object.add(veins);return;}
  const count=name.includes('thorax')?170:name.includes('head')?80:name.includes('abdomen')?35:/tibia|trochanterfemur/.test(name)?14:0;
  if(!count)return;
  const hairs=new THREE.InstancedMesh(new THREE.ConeGeometry(1,1,3),new THREE.MeshStandardMaterial({color:'#292016',roughness:.95}),count);
  hairs.name='bristles';const tip=new THREE.Object3D(),position=new THREE.Vector3(),normal=new THREE.Vector3(),up=new THREE.Vector3(0,1,0);
  for(let i=0;i<count;i++){
    const vertex=Math.floor(random()*positions.count),length=(.045+random()*.06)*(name.includes('nmf/c_')?1:.55);
    position.fromBufferAttribute(positions,vertex);normal.fromBufferAttribute(normals,vertex).normalize();
    tip.position.copy(position).addScaledVector(normal,length/2);tip.quaternion.setFromUnitVectors(up,normal);tip.scale.set(.002,length,.002);tip.updateMatrix();hairs.setMatrixAt(i,tip.matrix);
  }
  object.add(hairs);
}
