# Shakespeare Text Generation using RNN and LSTM
본 프로젝트에서는 RNN과 LSTM 모델을 활용하여 셰익스피어 문체의 텍스트를 생성하는 문자 단위 언어 모델을 구현함 
PyTorch를 사용해 Vanilla RNN과 LSTM 구조를 직접 설계하고, 두 모델의 언어 패턴 학습 성능을 비교.
LSTM 모델은 RNN보다 더 낮은 (Validation Loss을 보여, Long-term dependency을 더 잘 학습하는 것으로 나타남.
또한 Temperature 샘플링 기법을 적용하여 문법적 정확성과 창의성 간의 균형을 조정하였으며, temperature 0.7~1.0 구간에서 셰익스피어 특유의 문체를 가장 잘 재현하는 결과를 얻음.
이 프로젝트를 통해 시퀀스 모델링, 신경망 학습, 텍스트 생성의 핵심 원리를 이해할 수 있었음.
사용 기술: Python, PyTorch, Deep Learning, NLP, RNN, LSTM

# 프로젝트 구조
<h4>- main.py: RNN, LSTM 모델 학습 및 평가 검증 로직이 들어있는 스크립트</h4>
<h4>- generate.py: 학습된 모델을 사용하여 텍스트를 생성하는 스크립트</h4>
<h4>- model.py: Vanilla RNN & LSTM 모델을 정의하는 스크립트</h4>
<h4>- dataset.py: Shakespear 데이터셋을 로드하고 전처리하는 스크립트</h4>
<h4>- shakespeare_train.txt: 학습에 사용된 셰익스피어의 글이 포함된 데이터셋</h4>

## RNN Training & Validiation Loss plot 
<div style="display: flex; justify-content: space-between;">
    <img src="https://github.com/daewook1004/DeepLearning_LangugeModel/assets/113410302/d0f9ee37-4fdd-4ab2-9d1b-9a5b9a62b5e4" alt="Image 1" width="100%">
    <img src="https://github.com/daewook1004/DeepLearning_LangugeModel/assets/113410302/58bca3e4-3086-44db-962d-21008860b191" alt="Imag 2" width="100%">
</div>
<h3>- Training과 Validation의 loss값 모두 20 epoch까지 안정적으로 감소함</h3>

## LSTM Training & Validiation Loss plot 
<div style="display: flex; justify-content: space-between;">
    <img src="https://github.com/daewook1004/DeepLearning_LangugeModel/assets/113410302/b8011d8c-0520-4709-ba92-473a4db44605" alt="Image 1" width="100%">
    <img src="https://github.com/daewook1004/DeepLearning_LangugeModel/assets/113410302/0b6d6681-91bf-4978-91a0-20e463976c73" alt="Image 2" width="100%">
</div>
<h3>- Training과 Validation의 loss값 모두 20 epoch까지 안정적으로 감소함 </h3>

## Training Parameters
<h4> - sequence_length = 30 </h4>
<h4> - batch_size      = 64 </h4>
<h4> - n_layers        = 2  </h4>
<h4> - hidden_size     = 128 </h4>
<h4> - dropout         = 0.3 </h4>
<h4> - num_epochs      = 20  </h4>
<h4> - learning_rate   = 0.0001 </h4>

## Comparison between RNN and LSTM
<h4> Epoch 1/20, RNN:  Training Loss: 2.2198, Validation Loss: 1.9893 </h4>
<h4> Epoch 1/20, LSTM: Training Loss: 2.3715, Validation Loss: 2.0325 </h4>
...
<h4> Epoch 20/20, RNN:  Training Loss: 1.5366, Validation Loss: 1.6502 </h4>
<h4> Epoch 20/20, LSTM: Training Loss: 1.3779, Validation Loss: 1.5960 </h4>

<h3>- 두 모델 모두 20 epoch까지 양호한 일반화 성능을 보여줌 </h3>
<h3> - LSTM 모델이 RNN보다 더 우수한 성능, 더 낮은 Training & Validation Loss 값을 기록함 </h3>

## Generate characters with Best trained Model(LSTM) - Parameters
<h4> seed_char:  ['F', 'K', 'N', 'S', 'U'] </h4>
<h4> temperatures = [0.3, 0.5, 0.7, 1.0, 1.5]  * 0.3 미만이면 생성 x </h4>

## Generated Text Example
<h4>Temperature: 0.3</h4>

<h5>Seed Character: F  --------------     First Senator: What they have been the forth of the people, That we will not be the country to the pe</h5> 
<h5>Seed Character: K  --------------     KINGHAM: How now the senate, and the people. First Senator: So, the gods be many dear with him.</h5>
<h5>Seed Character: N  --------------     NENIUS: We have been the gods shall be somether send the people, and his country's son, That have not</h5>
<h5>Seed Character: S  --------------     S: The common her with a man of the world be so dear means and his father should be thee.</h5>
<h5>Seed Character: U  --------------     US: I were your heart, that which they are a senators, I will not be the people.</h5>

<h4>Temperature: 0.5</h4>

<h5>Seed Character: F  --------------     First Senator: How are not be to the tribunes Like in the sun, and the state of the poor price before</h5> 
<h5>Seed Character: K  --------------     KINGLADY ANNE: What is the city to fear of the other To be remain in the gods and death, and the hatt</h5>
<h5>Seed Character: N  --------------     NIUS: The gods did a good could to do the streat, Which they have peace. CORIOLANUS: How do you the </h5>
<h5>Seed Character: S  --------------     S: The general cause of strike at the poor unmented so devil to be content That stay me the same of t</h5>
<h5>Seed Character: U  --------------     US: Why, what you have been the mother? MARCIUS: I will be revenged me to whom you have deeds</h5>


<h4>Temperature: 0.7</h4>

<h5>Seed Character: F  --------------     From me but to power we have too the people; and excoes me? SICINIUS: Go he lay your long.</h5> 
<h5>Seed Character: K  --------------     KINGHAM: Ke more for the people do you do you done: The noble companion that cry show the first buttervingman;</h5>
<h5>Seed Character: N  --------------     NIUS: I will and before them for they are To great bears and his whole hearts, where in been by for t</h5>
<h5>Seed Character: S  --------------     S: A will be speak to do not the happy alone. SICINIUS: You have remain my true speak the dather.</h5>
<h5>Seed Character: U  --------------     US: Do not to back of these state, I have been one to ears, my part of mercy, By such that you made t</h5>

<h4>Temperature: 1.0</h4>

<h5>Seed Character: F  --------------     First Lords: Fie, done you thy country. BRUTUS: The name of thy such fonsolets: What makes me but am</h5> 
<h5>Seed Character: K  --------------     Kiend all me, sir, which at your heart Than parsing the rock that the ragh, And two be virtue the dic</h5>
<h5>Seed Character: N  --------------     NIUS: It deservous suelingment are Hath briefst I have but Corioli me that you have pardon: Standing,</h5>
<h5>Seed Character: S  --------------     S: Our award. Second Servingman: You say serve comes death with shown myself? And spir of yourselves </h5>
<h5>Seed Character: U  --------------     UFIDIUS: Should yet you As I can less my bond my chance Frame? our country's end Bratthe thine this, </h5>


<h4>Temperature: 1.5</h4>

<h5>Seed Character: F  --------------     F'd of. Which anstand you? You, grannevew bodeved! Towarls: and Tewly Put thy dix, getchilargunate, o</h5> 
<h5>Seed Character: K  --------------     Kt I, to to go.; dediment. Sull us. LACTIUS: Propeht, to be now? MENENIUS: I I'll stative Whyerfest </h5>
<h5>Seed Character: N  --------------     Nd, tew waoldren make my isomplanown Co i' their knowy in do in You; if;, Besterly, may, had's royhly</h5>
<h5>Seed Character: S  --------------     Second Citizen: Ye, Tower, you quayst King. Delives? Colome, From these scunce-falder blood MA'Rire:</h5>
<h5>Seed Character: U  --------------     US: Thou! looktare myou, Motaltewive, sirs-nris'lls, writy.n Like noble madam Wife methools led fool </h5>

## Discussion 
- Temperature가 낮을 수록 글의 문법이 더 정확하고 알기 쉽지만 글의 창의성이 부족
- Temperature가 높을 수록 다양한 단어를 활용하여 창의성이 높지만 문법이나 형식이 오류가 있음
- 0.7~1.0사이의 Temperature가 shakespeare의 풍의 재현하여 생성하는 데 적합하다고 생각함 
