# Changelog

## [3.1.0](https://github.com/nomic-ai/nomic/compare/v3.0.44...v3.1.0) (2024-07-19)


### Features

* add option for nomic-project-v2 ([857e6c4](https://github.com/nomic-ai/nomic/commit/857e6c489a6c6b9b639d18e58fb39b09ed30da22))
* allow arrow table for upload in map_text ([2f0007a](https://github.com/nomic-ai/nomic/commit/2f0007a02561e23b5ee4451310eb5a9b5fb3e371))
* classifier example ([#321](https://github.com/nomic-ai/nomic/issues/321)) ([1d350bd](https://github.com/nomic-ai/nomic/commit/1d350bd8e553148aad962e1126a0efff75715996))
* emb for client ([25c5b5e](https://github.com/nomic-ai/nomic/commit/25c5b5e9b7c634feabe83f308f5449c605a89b56))
* img embed updates ([31ab403](https://github.com/nomic-ai/nomic/commit/31ab40365fd50f7a4cc82c571cf0d0d654a72657))
* map_data support for Nomic Embed Vision ([#308](https://github.com/nomic-ai/nomic/issues/308)) ([e4a1c73](https://github.com/nomic-ai/nomic/commit/e4a1c73964843619ff964ef19e019e9076ab5886))
* nomic sagemaker vision ([2bda64f](https://github.com/nomic-ai/nomic/commit/2bda64f57af99adcd307f4ddfc86a309629b532a))
* notebook updates ([4a1f888](https://github.com/nomic-ai/nomic/commit/4a1f888fc4c9fd6dbf8b0de5e5b7c088716c2103))
* run black & isort ([00f5e48](https://github.com/nomic-ai/nomic/commit/00f5e48e9bd230f571bb360a497ef487291adb79))
* sagemaker client updates for batched image ([#319](https://github.com/nomic-ai/nomic/issues/319)) ([6c3d91e](https://github.com/nomic-ai/nomic/commit/6c3d91ed723eddc0dbab5bb7ee3d0de8dbcfcf68))
* task_type ([cff0e2b](https://github.com/nomic-ai/nomic/commit/cff0e2bc2042346cfd5b1f7c9f85890b7e05b45b))
* update local neighborhood parameter name and make all default ([eaba319](https://github.com/nomic-ai/nomic/commit/eaba319fc48e7ccdd588d4826362312871bec816))


### Bug Fixes

* allow indexed_field is none for image datasets with `create_index` ([#315](https://github.com/nomic-ai/nomic/issues/315)) ([2d7d2da](https://github.com/nomic-ai/nomic/commit/2d7d2dab72e709440ce1987810bd1da0ca2b1471))
* api naming consistency, version ([8b9d37a](https://github.com/nomic-ai/nomic/commit/8b9d37accdb100fc7aa9cd645bb501acc4905f99))
* assert image right type ([469f44a](https://github.com/nomic-ai/nomic/commit/469f44a8b897013d0dd6ca7b733c7b62aa8cde1f))
* bugs introduced after adding types ([56d298e](https://github.com/nomic-ai/nomic/commit/56d298e47d229dec24f2a0fbbc597afb2648f329))
* change file name ([552ec75](https://github.com/nomic-ai/nomic/commit/552ec75e971ba8c915e44b2bee00e5685a9fc2df))
* check format in nomic/ folder ([d96ca3a](https://github.com/nomic-ai/nomic/commit/d96ca3a83fede438b0230b956e0289545be747ed))
* dataset sidecar download - datum id sidecars are special ([24e7d94](https://github.com/nomic-ai/nomic/commit/24e7d949dbba4be5beb995f3e8dd3301d3905779))
* don't use b64 encode ([338a5f4](https://github.com/nomic-ai/nomic/commit/338a5f4353cdcd360c4651b67caa2bc5ac43d946))
* fetch db-registered topic sidecars ([df8a75a](https://github.com/nomic-ai/nomic/commit/df8a75aa8740db3b33974dd3277124af48bd537d))
* max image request ([41c459d](https://github.com/nomic-ai/nomic/commit/41c459d0891b7cdb6260be30defd87d14e508861))
* move indexed_field check earlier for image dataset ([#320](https://github.com/nomic-ai/nomic/issues/320)) ([4b93268](https://github.com/nomic-ai/nomic/commit/4b932680e02ce93ae015ec1a9df132e679495c93))
* nullable parameter ([f20f7c9](https://github.com/nomic-ai/nomic/commit/f20f7c9387060baf98f58e5908dd532020b4ac4f))
* outofbounds day ([5f17f95](https://github.com/nomic-ai/nomic/commit/5f17f95aff74f5658ba67800959cb48e326625b6))
* outofbounds day ([43da768](https://github.com/nomic-ai/nomic/commit/43da7685dd8442fe1a1c1407e4c5de8e705c333c))
* parsing problem ([0fdc178](https://github.com/nomic-ai/nomic/commit/0fdc178ab120ca221418253c1af33047961f8aec))
* Path and Tuple type isssues ([75f7767](https://github.com/nomic-ai/nomic/commit/75f7767b290b8e111757f0b777617ba2376636bf))
* problems with pa.compute ([1c3a8ef](https://github.com/nomic-ai/nomic/commit/1c3a8ef8592cc7727a6163aaf723a1f8896986c4))
* remove libcairo ([033c894](https://github.com/nomic-ai/nomic/commit/033c894c14c92532a9946c4e665259a5ccdd0774))
* remove pdb ([f8d2050](https://github.com/nomic-ai/nomic/commit/f8d20508ba90bde8b3ef2f16f99fec2bae499edd))
* resizing logic bug ([#306](https://github.com/nomic-ai/nomic/issues/306)) ([11ab3c5](https://github.com/nomic-ai/nomic/commit/11ab3c5d6433adebb3b5ecbcfd8e343373c089c3))
* respect modality in create_index ([#317](https://github.com/nomic-ai/nomic/issues/317)) ([fd3c108](https://github.com/nomic-ai/nomic/commit/fd3c108925288e02d2be3cb13abb48ff0c3fb4d0))
* return model name, not str model ([c9f9703](https://github.com/nomic-ai/nomic/commit/c9f970364d13e531bc2c27b8c275e835b57e8302))
* spelling ([d0a3ce5](https://github.com/nomic-ai/nomic/commit/d0a3ce5b0ebf412d7c17afa94bebd140e4bab6f8))
* text mode truncation param ([f6572ac](https://github.com/nomic-ai/nomic/commit/f6572ac6ae645839a01023dfa14a566aea4e2096))
* topic label field default to indexed field if not supplied ([#316](https://github.com/nomic-ai/nomic/issues/316)) ([8bc157d](https://github.com/nomic-ai/nomic/commit/8bc157db8d307b403b8d301565de2ca1fda46648))
* type issues after rebasing ([12c1b08](https://github.com/nomic-ai/nomic/commit/12c1b08e9625b60857609a8652bfca12b91660cb))
* typing ([8086591](https://github.com/nomic-ai/nomic/commit/80865918677a472443aebda1642d057a9ea500d6))
* typing + resize from file ([ef9703a](https://github.com/nomic-ai/nomic/commit/ef9703a223adb1122f91627a46db6a6c976f052d))
* update example ([2681a66](https://github.com/nomic-ai/nomic/commit/2681a667fca739f9bfe7728def3963cf5ca7c7dc))
* update min dim ([35a491d](https://github.com/nomic-ai/nomic/commit/35a491de1de769797ceb3d987728e7a14168c50c))
* use Optional ([d4d5eb3](https://github.com/nomic-ai/nomic/commit/d4d5eb3d046afc7ae1ba6763a2dd3ad6f430bb7e))
* wait for project log ([9cacd4a](https://github.com/nomic-ai/nomic/commit/9cacd4a5ff0fe1cc077cd57e2d7e7297ed399d66))


### Documentation

* update docs to make more clear that blobs are stored locally only ([#318](https://github.com/nomic-ai/nomic/issues/318)) ([aa863ec](https://github.com/nomic-ai/nomic/commit/aa863ecac7c315398559d9c8d7f5a558b2ca8837))
