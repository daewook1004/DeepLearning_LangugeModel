# Shakespeare Text Generation using RNN and LSTM

This project implements a **character-level language model** using **RNN** and **LSTM** to generate text in the style of Shakespeare.  
Using **PyTorch**, both Vanilla RNN and LSTM architectures were designed and compared to evaluate their ability to learn linguistic patterns.
The LSTM model achieved **lower validation loss**, showing better performance in learning **long-term dependencies**.  
Additionally, **temperature sampling** was applied to balance grammatical accuracy and creativity.  
Texts generated with temperatures between **0.7–1.0** most effectively reproduced Shakespearean linguistic style.

---

## Technologies Used
**Python**, **PyTorch**, **Deep Learning**, **NLP**, **RNN**, **LSTM**

---

## 📂 Project Structure
```
├── main.py # Training and validation logic for RNN & LSTM models
├── generate.py # Generates text using the trained model
├── model.py # Defines Vanilla RNN & LSTM architectures
├── dataset.py # Loads and preprocesses the Shakespeare dataset
└── shakespeare_train.txt # Source text data for training
```
---
## 📉 RNN Training & Validation Loss
<div align="center">
    <img src="https://github.com/daewook1004/DeepLearning_LangugeModel/assets/113410302/d0f9ee37-4fdd-4ab2-9d1b-9a5b9a62b5e4" width="45%">
    <img src="https://github.com/daewook1004/DeepLearning_LangugeModel/assets/113410302/58bca3e4-3086-44db-962d-21008860b191" width="45%">
</div>
Both training and validation loss steadily decreased across 20 epochs.

---
## 📉 LSTM Training & Validation Loss
<div align="center">
    <img src="https://github.com/daewook1004/DeepLearning_LangugeModel/assets/113410302/b8011d8c-0520-4709-ba92-473a4db44605" width="45%">
    <img src="https://github.com/daewook1004/DeepLearning_LangugeModel/assets/113410302/0b6d6681-91bf-4978-91a0-20e463976c73" width="45%">
</div>

✅ The LSTM model showed smoother convergence and lower overall loss compared to RNN.
## ⚙️ Training Parameters

| Parameter        | Value  |
|------------------|--------|
| sequence_length  | 30     |
| batch_size       | 64     |
| n_layers         | 2      |
| hidden_size      | 128    |
| dropout          | 0.3    |
| num_epochs       | 20     |
| learning_rate    | 0.0001 |

---

## 📊 Model Performance Comparison

| Epoch | Model | Training Loss | Validation Loss |
|:------:|:------:|:---------------:|:----------------:|
| 1 | RNN  | 2.2198 | 1.9893 |
| 1 | LSTM | 2.3715 | 2.0325 |
| 20 | RNN  | 1.5366 | 1.6502 |
| 20 | LSTM | 1.3779 | 1.5960 |

> 🔹 **LSTM outperformed RNN**, achieving lower training and validation loss and demonstrating stronger long-term dependency learning.

---

## ✍️ Text Generation (Best Model: LSTM)

### 🔧 Parameters
- **Seed Characters:** `['F', 'K', 'N', 'S', 'U']`
- **Temperatures:** `[0.3, 0.5, 0.7, 1.0, 1.5]` (No generation below 0.3)

---
### 🧾 Example Outputs

#### Temperature = 0.3
> Grammatically consistent but low creativity.
<h5>Seed Character: F  --------------     First Senator: What they have been the forth of the people, That we will not be the country to the pe</h5> 
<h5>Seed Character: K  --------------     KINGHAM: How now the senate, and the people. First Senator: So, the gods be many dear with him.</h5>
<h5>Seed Character: N  --------------     NENIUS: We have been the gods shall be somether send the people, and his country's son, That have not</h5>
<h5>Seed Character: S  --------------     S: The common her with a man of the world be so dear means and his father should be thee.</h5>
<h5>Seed Character: U  --------------     US: I were your heart, that which they are a senators, I will not be the people.</h5>

#### Temperature = 0.7
> Balanced between structure and creativity; closely resembles Shakespeare’s tone.

<h5>Seed Character: F  --------------     From me but to power we have too the people; and excoes me? SICINIUS: Go he lay your long.</h5> 
<h5>Seed Character: K  --------------     KINGHAM: Ke more for the people do you do you done: The noble companion that cry show the first buttervingman;</h5>
<h5>Seed Character: N  --------------     NIUS: I will and before them for they are To great bears and his whole hearts, where in been by for t</h5>
<h5>Seed Character: S  --------------     S: A will be speak to do not the happy alone. SICINIUS: You have remain my true speak the dather.</h5>
<h5>Seed Character: U  --------------     US: Do not to back of these state, I have been one to ears, my part of mercy, By such that you made t</h5>

#### Temperature = 1.0 
> Highly creative but grammatically inconsistent.

<h5>Seed Character: F  --------------     First Lords: Fie, done you thy country. BRUTUS: The name of thy such fonsolets: What makes me but am</h5> 
<h5>Seed Character: K  --------------     Kiend all me, sir, which at your heart Than parsing the rock that the ragh, And two be virtue the dic</h5>
<h5>Seed Character: N  --------------     NIUS: It deservous suelingment are Hath briefst I have but Corioli me that you have pardon: Standing,</h5>
<h5>Seed Character: S  --------------     S: Our award. Second Servingman: You say serve comes death with shown myself? And spir of yourselves </h5>
<h5>Seed Character: U  --------------     UFIDIUS: Should yet you As I can less my bond my chance Frame? our country's end Bratthe thine this, </h5>

#### Temperature = 1.5
<h5>Seed Character: F  --------------     F'd of. Which anstand you? You, grannevew bodeved! Towarls: and Tewly Put thy dix, getchilargunate, o</h5> 
<h5>Seed Character: K  --------------     Kt I, to to go.; dediment. Sull us. LACTIUS: Propeht, to be now? MENENIUS: I I'll stative Whyerfest </h5>
<h5>Seed Character: N  --------------     Nd, tew waoldren make my isomplanown Co i' their knowy in do in You; if;, Besterly, may, had's royhly</h5>
<h5>Seed Character: S  --------------     Second Citizen: Ye, Tower, you quayst King. Delives? Colome, From these scunce-falder blood MA'Rire:</h5>
<h5>Seed Character: U  --------------     US: Thou! looktare myou, Motaltewive, sirs-nris'lls, writy.n Like noble madam Wife methools led fool </h5>

# 💬 Discussion
- **Low Temperature (0.3–0.5):** High grammatical accuracy, low creativity.  
- **High Temperature (1.0–1.5):** High creativity, frequent grammatical errors.  
- **Optimal Range (0.7–1.0):** Most natural and expressive Shakespearean-style text generation.

> The LSTM model with temperature values between **0.7–1.0** best reproduces the poetic and rhythmic qualities of Shakespeare’s writing.
