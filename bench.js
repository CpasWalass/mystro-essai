// Essai B : temps d'exécution des modèles de homr avec ONNX Runtime Web (WebAssembly).
// Les entrées sont factices mais ont les formes réelles mesurées sur test7 (voir ../profile_homr.py) :
// le temps de calcul d'un réseau ne dépend pas des valeurs des pixels.
/* global ort */

const PATCHES = 54 // morceaux 320×320 passés à la segmentation pour une page
const STAVES = 4 // portées de test7
const DECODER_STEPS = 130 // symboles décodés par portée (522 appels / 4 portées)
const STAFF_SAMPLE = 1 // portées réellement calculées ; le résultat est extrapolé à STAVES

const logElement = document.getElementById('log')
const log = (line) => {
  logElement.textContent += `${line}\n`
  console.log(line)
}
const seconds = (ms) => `${(ms / 1000).toFixed(1)} s`
const memory = () =>
  performance.memory ? `${Math.round(performance.memory.usedJSHeapSize / 1e6)} Mo (tas JS)` : 'non mesurable'

async function timed(label, action) {
  const start = performance.now()
  const result = await action()
  const elapsed = performance.now() - start
  log(`${label} : ${seconds(elapsed)}`)
  return { result, elapsed }
}

function float32(dims, fill = 0.5) {
  return new ort.Tensor('float32', new Float32Array(dims.reduce((a, b) => a * b, 1)).fill(fill), dims)
}

function int64(value) {
  return new ort.Tensor('int64', BigInt64Array.from([BigInt(value)]), [1, 1])
}

async function runSegmentation(session) {
  const input = float32([1, 3, 320, 320], 1)
  for (let i = 0; i < PATCHES; i++) await session.run({ input })
}

async function runStaff(encoder, decoder) {
  const { output: context } = await encoder.run({ input: float32([1, 1, 256, 1280]) })
  const contextReduced = new ort.Tensor('float32', context.data.slice(0, 512), [1, 1, 512])
  const cacheNames = decoder.inputNames.filter((name) => name.startsWith('cache_in'))
  let cache = cacheNames.map(() => new ort.Tensor('float32', new Float32Array(0), [1, 8, 0, 64]))

  for (let step = 0; step < DECODER_STEPS; step++) {
    const feeds = {
      rhythms: int64(1),
      pitchs: int64(1),
      lifts: int64(1),
      articulations: int64(1),
      slurs: int64(1),
      context: step === 0 ? context : contextReduced,
      cache_len: new ort.Tensor('int64', BigInt64Array.from([BigInt(step)]), [1]),
    }
    cacheNames.forEach((name, i) => (feeds[name] = cache[i]))
    const outputs = await decoder.run(feeds)
    // Comme homr : les sorties 7 et suivantes sont le cache de l'étape suivante.
    cache = decoder.outputNames.slice(7).map((name) => outputs[name])
  }
}

async function main() {
  const button = document.getElementById('start')
  button.disabled = true
  // ?threads=1 force un seul fil ; ?int8=1 charge les modèles quantifiés (spikes/phase-b, quantize_dynamic).
  const params = new URLSearchParams(location.search)
  const maxThreads = Number(params.get('threads') ?? 4)
  const variant = '.int8' // seuls les modèles quantifiés sont publiés sur ce site d'essai
  const threads = self.crossOriginIsolated ? Math.min(maxThreads, navigator.hardwareConcurrency || 1) : 1
  ort.env.wasm.numThreads = threads
  ort.env.wasm.wasmPaths = new URL('ort/', location.href).href
  log(`Navigateur : ${navigator.userAgent}`)
  log(`Processeurs : ${navigator.hardwareConcurrency}, fils de calcul utilisés : ${threads}` +
      (self.crossOriginIsolated ? '' : ' (page non isolée : un seul fil possible)'))
  log(`Modèles : ${variant ? 'quantifiés 8 bits' : 'originaux'}`)
  log(`Mémoire de l'appareil (approx.) : ${navigator.deviceMemory ?? '?'} Go`)

  try {
    const options = { executionProviders: ['wasm'], graphOptimizationLevel: 'all' }
    const { result: segnet } = await timed('Chargement segnet', () => ort.InferenceSession.create(`models/segnet${variant}.onnx`, options))
    const { result: encoder } = await timed('Chargement encodeur', () => ort.InferenceSession.create(`models/encoder${variant}.onnx`, options))
    const { result: decoder } = await timed('Chargement décodeur', () => ort.InferenceSession.create(`models/decoder${variant}.onnx`, options))
    log(`Mémoire après chargement : ${memory()}`)

    const { elapsed: segmentation } = await timed(`Segmentation (${PATCHES} morceaux)`, () => runSegmentation(segnet))
    let staffTime = 0
    for (let i = 0; i < STAFF_SAMPLE; i++) {
      staffTime += (await timed(`Portée ${i + 1} (encodeur + ${DECODER_STEPS} pas de décodeur)`, () => runStaff(encoder, decoder))).elapsed
    }
    const page = segmentation + (staffTime / STAFF_SAMPLE) * STAVES
    log(`Mémoire après calcul : ${memory()}`)
    log(`\nESTIMATION POUR UNE PAGE (${STAVES} portées) : ${seconds(page)}`)
    log('(+ environ 2 s de traitement d\'image sur PC, davantage sur téléphone)')
  } catch (error) {
    log(`ÉCHEC : ${error?.message ?? error}`)
  } finally {
    button.disabled = false
  }
}

document.getElementById('start').addEventListener('click', main)
