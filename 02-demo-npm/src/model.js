import * as tf from '@tensorflow/tfjs'

export const train = () => {
  tf.tidy(() => {
    run()
  })
}

const readRawData = (csvUrl) => {
  const readData = tf.data.csv(csvUrl, {
    columnConfigs: {
      toxic: {
        isLabel: true
      }
    }
  })

    return readData
}

const run = async () => {
  const rawDataResult = readRawData('data/toxic_data_sample.csv')

  await rawDataResult.forEachAsync(row => {
    console.log(row)
  })
}
