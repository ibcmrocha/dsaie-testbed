/* Frozen 2 → 64 → 784 ReLU/sigmoid VAE decoder, exported by train_mnist_vae.py.
   No tensor library, GPU, server inference, or atlas interpolation is involved. */
(() => {
  'use strict';
  function float16(bits) {
    const sign=(bits&0x8000)?-1:1,exponent=(bits>>>10)&31,mantissa=bits&1023;
    if(exponent===0)return sign*mantissa*2**-24;
    if(exponent===31)return mantissa?NaN:sign*Infinity;
    return sign*(1+mantissa/1024)*2**(exponent-15);
  }
  function unpack(buffer,index,model) {
    const count=model.weightCount,view=new DataView(buffer,index*count*2,count*2),weights=new Float32Array(count);
    for(let i=0;i<count;i++)weights[i]=float16(view.getUint16(i*2,true));
    const h=model.hidden;
    return {hidden:h,w1:weights.subarray(0,2*h),b1:weights.subarray(2*h,3*h),
      w2:weights.subarray(3*h,3*h+784*h),b2:weights.subarray(3*h+784*h),
      activation:new Float32Array(h),pixels:new Float32Array(784)};
  }
  function decode(network,z) {
    const {hidden,w1,b1,w2,b2,activation,pixels}=network;
    for(let i=0;i<hidden;i++)activation[i]=Math.max(0,b1[i]+w1[2*i]*z[0]+w1[2*i+1]*z[1]);
    for(let i=0;i<784;i++){
      let sum=b2[i],offset=i*hidden;
      for(let j=0;j<hidden;j++)sum+=w2[offset+j]*activation[j];
      pixels[i]=1/(1+Math.exp(-sum));
    }
    return pixels;
  }
  window.MnistVae={unpack,decode};
})();
