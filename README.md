# Ex Anima Core

Ex Anima Core는 C++ 프로그램에 내장하는 범용 행동 모방학습 라이브러리다. 첫 알고리즘은 Behavioral Cloning이다. 신경망을 자체 구현하고 행렬 연산에는 Eigen을 사용할 수 있다.

## 모듈 책임

| 모듈 | 책임 |
| --- | --- |
| Neural Core | 모델 구조, 계층, 활성화 함수, 수치 연산과 순전파 |
| Training | 손실 함수, 역전파와 옵티마이저 |
| Inference | 학습 기능 없이 빌드할 수 있는 모델 로드 및 추론 |
| Imitation | 시연 데이터와 Behavioral Cloning |

호출자가 입력 특성 순서와 출력 라벨을 정의한다. JSON 설정으로 모델 구조와 과거 상태 사용 여부를 표현한다. 초기 출력은 단일 선택이다. 모델 구조와 메타데이터는 JSON, 가중치와 편향은 Binary로 저장한다.

## 초기 범위 제안

- CPU 기반 Dense MLP, ReLU, 수치적으로 안정적인 Softmax 및 다중 클래스 Cross Entropy, SGD와 Adam.
- 단일 선택 Behavioral Cloning, 선택적 프레임 스태킹, 모델 저장·로드와 파라미터·활성화 조회.
- Gradient Checking, 참조 수치 비교, 독립 추론 빌드와 저장 전후 출력 일치 검증.

Core는 다른 Ex Anima 저장소와 게임 엔진에 의존하지 않도록 설계한다. 학습은 로컬 오프라인에서 수행한다. C++ 표준과 공개 API, CMake 대상명, JSON 및 Binary 세부 규격은 상세 설계에서 결정한다.

## 최소 학습·추론 실행

C++17과 CMake 3.20 이상이 필요하다. CMake는 SHA-256을 고정한 nlohmann/json 3.12.0을 사용한다. 동일한 공식 소스 묶음의 로컬 경로는 `EXANIMA_JSON_ARCHIVE`로 지정할 수 있다.

```powershell
cmake -S . -B build/full
cmake --build build/full --config Release
ctest --test-dir build/full -C Release --output-on-failure
cmake -S . -B build/inference -DEXANIMA_BUILD_TRAINING=OFF -DBUILD_TESTING=OFF
cmake --build build/inference --config Release
```

`exanima-train`은 현재 상태 벡터에서 하나의 행동을 예측하는 ReLU MLP를 SGD로 학습한다. `exanima-infer`는 JSON+Binary 모델을 로드해 관측별 인덱스, 라벨과 점수를 출력한다. 파라미터는 API에서 조회할 수 있다.

[C++ API와 모델 형식](https://github.com/ex-anima/docs/blob/feature/sandbox-first/CORE_OFFLINE_BC.md) · [통합 프로젝트 기획서 v0.2](https://github.com/ex-anima/docs/blob/main/PROJECT_PLAN.md)
