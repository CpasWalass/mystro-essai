# Essai provisoire : reconnaissance de partitions dans le navigateur

Page de mesure de performance pour MyStroAi : exécute dans le navigateur, avec ONNX Runtime Web,
les modèles de reconnaissance de partitions de [homr](https://github.com/liebharc/homr) sur des
entrées factices, et chronomètre le calcul d'une page.

Ce dépôt est temporaire et sera supprimé après les mesures.

## Licences des composants

- `models/` : modèles de [homr](https://github.com/liebharc/homr) (AGPL-3.0), poids quantifiés en 8 bits
  avec `onnxruntime.quantization.quantize_dynamic`, sans autre modification. Le code source de homr est
  disponible sur son dépôt.
- `ort/` : [ONNX Runtime Web](https://github.com/microsoft/onnxruntime) 1.23.2 (MIT).
- `coi-serviceworker.min.js` : [coi-serviceworker](https://github.com/gzuidhof/coi-serviceworker) 0.1.7 (MIT).
- `index.html`, `bench.js` : page d'essai, publiée sous AGPL-3.0.
