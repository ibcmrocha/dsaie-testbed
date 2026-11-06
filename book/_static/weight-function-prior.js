/* The same finite-rank Gaussian prior, sampled through two independent routes.
   No diagonal jitter: N > 5 gives a valid degenerate Gaussian, not extra noise. */
(() => {
  'use strict';
  const centers=[0,1.5,3,4.5,6];
  const basis=(x,j)=>Math.exp(-.5*(x-centers[j])**2);
  const dot=(a,b)=>a.reduce((sum,v,i)=>sum+v*b[i],0);
  function normal(){return Math.sqrt(-2*Math.log(Math.max(Math.random(),1e-15)))*Math.cos(2*Math.PI*Math.random());}
  function covarianceFactor(K){
    const n=K.length,A=K.map(row=>row.slice()),V=K.map((_,i)=>K.map((_,j)=>Number(i===j)));
    const scale=Math.max(1,...K.map((row,i)=>row[i]));
    // Symmetric Jacobi rotations preserve the null space of this PSD covariance.
    for(let step=0;step<100*n*n;step++){
      let p=0,q=1;
      for(let i=0;i<n;i++)for(let j=i+1;j<n;j++)if(Math.abs(A[i][j])>Math.abs(A[p][q])){p=i;q=j;}
      if(Math.abs(A[p][q])<=1e-13*scale)break;
      const angle=.5*Math.atan2(2*A[p][q],A[q][q]-A[p][p]),c=Math.cos(angle),s=Math.sin(angle);
      const app=A[p][p],aqq=A[q][q],apq=A[p][q];
      for(let k=0;k<n;k++)if(k!==p&&k!==q){
        const kp=A[k][p],kq=A[k][q];
        A[k][p]=A[p][k]=c*kp-s*kq;A[k][q]=A[q][k]=s*kp+c*kq;
      }
      A[p][p]=c*c*app-2*c*s*apq+s*s*aqq;
      A[q][q]=s*s*app+2*c*s*apq+c*c*aqq;A[p][q]=A[q][p]=0;
      for(let k=0;k<n;k++){
        const vp=V[k][p],vq=V[k][q];V[k][p]=c*vp-s*vq;V[k][q]=s*vp+c*vq;
      }
    }
    const eigenvalues=A.map((row,i)=>row[i]),tolerance=1e-11*Math.max(...eigenvalues);
    if(eigenvalues.some(v=>v < -tolerance))throw Error('The function covariance is not positive semidefinite.');
    const roots=eigenvalues.map(v=>v>tolerance?Math.sqrt(v):0);
    return {factor:V.map(row=>row.map((v,j)=>v*roots[j])),rank:roots.filter(v=>v>0).length};
  }
  function createPrior(n){
    if(!Number.isInteger(n)||n<2||n>20)throw RangeError('Choose 2 to 20 locations.');
    const xs=Array.from({length:n},(_,i)=>6*i/(n-1));
    const Phi=xs.map(x=>centers.map((_,j)=>basis(x,j))),K=Phi.map(a=>Phi.map(b=>dot(a,b)));
    const {factor,rank}=covarianceFactor(K);
    return {xs,Phi,K,factor,rank,
      sampleWeights(randomNormal=normal){const w=centers.map(()=>randomNormal());return {w,values:Phi.map(row=>dot(row,w))};},
      sampleFunctions(randomNormal=normal){const z=xs.map(()=>randomNormal());return {values:factor.map(row=>dot(row,z))};}
    };
  }
  window.WeightFunctionPrior={basis,covarianceFactor,createPrior};
})();
