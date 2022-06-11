const tf = require('@tensorflow/tfjs');
// require('@tensorflow/tfjs-node');

console.log(tf.version.tfjs)

async function loadApp () {
    await tf.ready()
    console.log(tf.getBackend())
}

loadApp()