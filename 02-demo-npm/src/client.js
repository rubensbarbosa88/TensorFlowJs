import * as tf from '@tensorflow/tfjs'
import '@tensorflow/tfjs-backend-wasm'

console.log(tf.version.tfjs)

async function loadApp () {
    tf.setBackend('wasm')
    await tf.ready()
    console.log(tf.getBackend())
}

loadApp()