import * as tf from '@tensorflow/tfjs'
// import '@tensorflow/tfjs-backend-wasm'
import * as model from './model'


// async function loadApp () {
//     tf.setBackend('wasm')
//     await tf.ready()
//     console.log(tf.getBackend())
// }

// loadApp()

// Tensor types
// tensor1d
// const age = tf.tensor1d([30, 25], 'int32')
// age.print()
// tf.print(age)
// console.log(age.shape)
// console.log(age.dtype)

// tensor2d
// const age_income_height = tf.tensor2d([[30, 1000, 170], [25, 2000, 168]])
// age_income_height.print()
// console.log(age_income_height.shape)
// console.log(age_income_height.dtype)

// scalar
// const multiplier = tf.scalar(10)
// multiplier.print()
// console.log(multiplier.dtype)

// Tensor Operations
// aditions
// const income_source1 = tf.tensor1d([100, 200, 300, 150])
// const income_source2 = tf.tensor1d([50, 70, 30, 20])
// const total_income =  tf.add(income_source1, income_source2)
// const total_income = income_source1.add(income_source2)
// total_income.print()

//variables


const init = async () => {
  await tf.ready()
  model.train()
}

init()
