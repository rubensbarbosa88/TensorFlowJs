import * as tf from '@tensorflow/tfjs'
import * as tfvis from '@tensorflow/tfjs-vis'

export const train = () => {
  tf.tidy(() => {
    run()
  })
}

const csvUrl = 'data/toxic_data_sample.csv'
const stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now']
let tmpDictionary = {}
let EMBEDDING_SIZE = 1000

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

const plotOutputLabelsCount = (labels) => {
  const labelCounts = labels.reduce((acc, label) => {
    acc[label] = acc[label] === undefined ? 1 : acc[label] += 1
    return acc
  }, {})

  const barChartData = []
  Object.keys(labelCounts).forEach(key => {
    barChartData.push({
      index: key,
      value: labelCounts[key]
    })
  })

  tfvis.render.barchart({
    name: 'Toxic output labels',
    tab: 'Exploration'
  }, barChartData)
}

const tokenize = (sentence, isCreateDict = false) => {
  const tmpTokens = sentence.split(/\s+/g)
  const tokens = tmpTokens.filter(token => !stopwords.includes(token) && token.length)

  if (isCreateDict) {
    tokens.reduce((acc, label) => {
      acc[label] = acc[label] === undefined ? 1 : acc[label] += 1
      return acc
    }, tmpDictionary)
  }

  return tmpTokens
}

const sortDictionaryByValue = (dict) => {
  const itens = Object.keys(dict).map(key => {
    return [key, dict[key]]
  })

  return itens.sort((a, b) => b[1] - a[1])
}


const getInverseDocumentFrequency = (documentTokens, dictionary) => {
  const getDocumentFrequency = (doc, token) => {
    return doc.reduce((acc, curr) => curr.includes(token) ? acc + 1 : acc, 0)
  }

  return dictionary.map(token => 1 + Math.log(documentTokens.length / getDocumentFrequency(documentTokens, token)))
}

const encoder = (sentence, dictionary, idfs) => {
  const tokens = tokenize(sentence)
  const tfs = getTermFrequency(tokens, dictionary)
  const tfidfs = getTfIdf(tfs, idfs)

  return tfidfs
}

const getTermFrequency = (tokens, dictionary) => {
  return dictionary.map(token => tokens.reduce((acc, curr) => {
    return curr == token ? acc + 1 : acc
  }, 0))
}

const getTfIdf = (tfs, idfs) => {
  return tfs.map((element, index) => element * idfs[index])
}

const prepareData = (dictionary, idfs) => {
  const preprocess = ({xs, ys}) => {
    const comment = xs.comment_text
    const trimmedComent = comment.toLowerCase().trim()
    const encoded = encoder(trimmedComent, dictionary, idfs)

    return {
      xs: tf.tensor2d([encoded], [1, dictionary.length]),
      ys: tf.tensor2d([ys.toxic], [1, 1])
    }
  }

  const readData = tf.data.csv(csvUrl, {
    columnConfigs: {
      toxic: {
        isLabel: true
      }
    }
  }).map(preprocess)

  return readData
}

const run = async () => {
  const rawDataResult = readRawData(csvUrl)
  const labels = []
  const comments = []
  const documentTokens = []

  await rawDataResult.forEachAsync(row => {
    const comment = row.xs.comment_text
    const trimmedComent = comment.toLowerCase().trim()

    comments.push(trimmedComent)
    documentTokens.push(tokenize(trimmedComent, true))
    labels.push(row.ys.toxic)
  })

  plotOutputLabelsCount(labels)

  const sortedTmpDictionary = sortDictionaryByValue(tmpDictionary)

  if (sortedTmpDictionary.length <= EMBEDDING_SIZE) {
    EMBEDDING_SIZE = sortedTmpDictionary.length
  }

  const dictionary = sortedTmpDictionary.slice(0, EMBEDDING_SIZE).map(row => row[0])

  const idfs = getInverseDocumentFrequency(documentTokens, dictionary)

  const ds = prepareData(dictionary, idfs)
  await ds.forEachAsync(e => console.log(e))
}
