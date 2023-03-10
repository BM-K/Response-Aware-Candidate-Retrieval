# KoBART-pytorch
๐ง KoBART summarization using pytorch + copy mechanism

## Data
- Data ๊ตฌ์กฐ
    - Train Data : 27,392
    - Valid Data : 6,850
    - Test Data : 8,561
 
## How to Train
- KoBART fine-tuning + Copy Mechanism
- Since the python libary was directly modified and used, I recommended to use a virtual environment. ๐
    - geneartion_utils.py -> /site-packages/transformers/_geneartion_utils.py_
    - modeling_bart.py -> /site-packages/transformers/models/bart/_modeling_bart.py_
- bash train_test.sh
```
[Training]
python train.py --train True --test False --batch_size 16 --max_len 512 --lr 5e-05 --epochs 10

[Testing-rouge]
python train.py --train False --test True --batch_size 16 --max_len 512
```

## Model Performance
- Test data's [rouge score](https://en.wikipedia.org/wiki/ROUGE_(metric)) 
### Base
| | rouge-1 |rouge-2|rouge-l|
|-------|--------:|--------:|--------:|
| Precision|0.5333|0.3463|0.4534|
| Recall|0.5775|0.3737|0.4869|
| F1|0.5381|0.3452|0.4555|

### Copy Mechanism
| | rouge-1 |rouge-2|rouge-l|
|-------|--------:|--------:|--------:|
| Precision|0.5698|0.3776|0.4882|
| Recall|0.5561|0.3612|0.4717|
| F1|0.5460|0.3545|0.4654|

## Examples
| | |Text|
|-------|:--------|:--------|
|1|๊ธฐ์ฌ|๊ฒฝ๊ธฐ๋์ ๊ฒฝ๊ธฐ๋์๊ณต์ฌ๋ ๊ด๊ต์์ฒ, ๋ํํธ์๊ณต์, ์ฑ๋จํ๊ต ๋ฑ 3๊ฐ ์ง๊ตฌ์ ๊ฑด๋ฆฝ ์์ ์ธ ์ด 730๊ฐ๊ตฌ์ ๊ฒฝ๊ธฐํ๋ณต์ฃผํ ์์ฃผ์๋ฅผ ๋ชจ์งํ๋ค๊ณ  26์ผ ๋ฐํ๋ค. ๋ชจ์ง ๊ธฐ๊ฐ์ ๋ค์๋ฌ 2์ผ๋ถํฐ 11์ผ๊น์ง์ด๋ฉฐ, โ๊ฒฝ๊ธฐ๋์๊ณต์ฌ ์๋์ฃผํ ์ฒญ์ฝ์ผํฐ(https://apply.gico.or.kr)โ์์ ์ธํฐ๋ท ์ฒญ์ฝ์ ์๋ก ์งํ๋๋ค. ๊ด๊ต์์ฒ ๊ฒฝ๊ธฐํ๋ณต์ฃผํ์ ์ ์ฉ๋ฉด์  16ใกํ ๋ํ์ 40๊ฐ๊ตฌ์ ์ฒญ๋ 20๊ฐ๊ตฌ, 26ใกํ ์ฒญ๋ 186๊ฐ๊ตฌ์ ๊ณ ๋ น์ 24๊ฐ๊ตฌ, ์ฃผ๊ฑฐ๊ธ์ฌ์๊ธ์ 30๊ฐ๊ตฌ๊น์ง ์ด 300๊ฐ๊ตฌ๋ฅผ ๋ชจ์งํ๋ค. ๋ณด์ฆ๊ธ 2์ฒ729๋ง4์ฒ&#126;4์ฒ783๋ง3์ฒ ์์ ์ ์๋๋ฃ๋ 11๋ง8์ฒ&#126;20๋ง7์ฒ ์์ด๋ค. ์์ฃผ ์์ ์ ์ค๋ 2020๋ 11์์ด๋ค. ๋ํํธ์๊ณต์ ๊ฒฝ๊ธฐํ๋ณต์ฃผํ์ ๋ํ2์ ๋์์ 6๊ฐ๋ 995๊ฐ๊ตฌ ์กฐ์ฑ๋๋ ๋๊ท๋ชจ ๋จ์ง์ด๋ค. ์ด๋ฒ ์์ฃผ์ ๋ชจ์ง์์๋ ๊ณต๊ธ๋ฉด์  44ใกํ ์ ํผ๋ถ๋ถ 130๊ฐ๊ตฌ๋ฅผ ์ฐ์  ๋ชจ์งํ๋ฉฐ ์๋์กฐ๊ฑด์ ๋ณด์ฆ๊ธ 5์ฒ๋ง ์์ ์ ์๋๋ฃ 20๋ง8์ฒ ์์ด๋ค. ๋ด๋ 12์ ์์ฃผ ์์ ์ผ๋ก, ๋๋จธ์ง ๊ฐ๊ตฌ๋ ์ฐ๋ง์ ๋ชจ์งํ  ์์ ์ด๋ค ์ฑ๋จํ๊ต ๊ฒฝ๊ธฐํ๋ณต์ฃผํ์ ์ ์ฉ๋ฉด์  16ใกํ ์ฐฝ์์ธ 100๊ฐ๊ตฌ์ ์ฒญ๋ 124๊ฐ๊ตฌ, 26ใกํ ์ฒญ๋ 46๊ฐ๊ตฌ์ ๊ณ ๋ น์ 30๊ฐ๊ตฌ ๋ฑ ์ด 300๊ฐ๊ตฌ๋ฅผ ๋ชจ์งํ๋ฉฐ, ๋ณด์ฆ๊ธ 3์ฒ876๋ง&#126;6์ฒ992๋ง ์์ ์ ์๋๋ฃ 14๋ง5์ฒ&#126;26๋ง2์ฒ ์์ด๋ค. ๊น์คํ ๋ ๋์์ฃผํ์ค์ฅ์ ""๋๋ด ์ฒญ๋์ธต์ ์ฃผ์ ๋์์ผ๋ก ํ ์ฃผ๊ฑฐ๋ณต์ง์ ์ฑ์ธ ๊ฒฝ๊ธฐํ๋ณต์ฃผํ์ 2022๋๊น์ง 1๋งํธ๋ฅผ ๊ณต๊ธํ๋ ๊ณผ์ ์์ ๋งค๋ ๊ณต๊ธ๋ฌผ๋์ด ๋์ด๋  ์์ ""์ด๋ผ๋ฉฐ ""๊ฒฝ๊ธฐํ๋ณต์ฃผํ ์ฌ์์๋ ๋ง์ ๊ด์ฌ์ ๊ฐ์ ธ๋ฌ๋ผ""๊ณ  ๋งํ๋ค.|
|1|๋ชจ๋ธ์์ฝ|๊ฒฝ๊ธฐ๋์ ๊ฒฝ๊ธฐ๋์๊ณต์ฌ๋ ๊ด๊ต์์ฒ, ๋ํํธ์๊ณต์, ์ฑ๋จํ๊ต ๋ฑ 3๊ฐ ์ง๊ตฌ์ ๊ฑด๋ฆฝ ์์ ์ธ ์ด 730๊ฐ๊ตฌ์ ๊ฒฝ๊ธฐํ๋ณต์ฃผํ ์์ฃผ์๋ฅผ ๋ชจ์งํ๋ค๊ณ  26์ผ ๋ฐํ์ผ๋ฉฐ ๋ชจ์ง ๊ธฐ๊ฐ์ ๋ค์๋ฌ 2์ผ๋ถํฐ 11์ผ๊น์ง์ด๋ฉฐ, ๋ณด์ฆ๊ธ 2์ฒ729๋ง4์ฒ&#126;4์ฒ783๋ง3์ฒ ์์ ์ ์๋๋ฃ๋ 11๋ง8์ฒ&#126;20๋ง7์ฒ ์์ด๋ค.|
|2|๊ธฐ์ฌ|์ ๋จ๊ฐ๋ฐ๊ณต์ฌ, โ์ผ์๋ฆฌ์ฐฝ์ถโ์ฐ์๊ธฐ๊ด ์ ์  ์ ๋จ๊ฐ๋ฐ๊ณต์ฌ ์ฒญ์ฌ ์ ๊ฒฝ. ์ ๋จ๊ฐ๋ฐ๊ณต์ฌ๊ฐ ์ผ์๋ฆฌ์ฐฝ์ถ ์ฐ์๊ธฐ๊ด์ ์ ์ ๋ผ ํ์ ์์ ๋ถ์ฅ๊ด ํ์ฐฝ์ ์์ํ๋ค. 6์ผ ์ ๋จ๊ฐ๋ฐฉ๊ณต์ฌ์ ๋ฐ๋ฅด๋ฉด ์ง๋ 3์ผ ์ธ์ข์ปจ๋ฒค์์ผํฐ์์ ์ด๋ฆฐ 2019๋ ์๋ฐ๊ธฐ ์ง๋ฐฉ๊ณต์ฌยท๊ณต๋จ CEO ๋ฆฌ๋์ญ ํฌ๋ผ์์ ์ด๊ฐ์ด ์์ ํ๋ค๊ณ  ๋ฐํ๋ค. ์ ๋จ๊ฐ๋ฐ๊ณต์ฌ๋ ๋ฌธ์ฌ์ธ ์ ๋ถ ์ญ์  ์ฌ์์ธ ์ผ์๋ฆฌ์ฐฝ์ถ ์ ์ฑ์ ์ ๊ทน ๋ถ์ํ๋ฉฐ, ์ง์ญ์ ๊ณ ์ฉ์์ฅ ํ์ฑํํ๊ธฐ ์ํด ์ ๊ท์ฌ์ ๋ฐ๊ตด ๋ฑ์ ์ญ์ ์ ๋๋ค. ์ด ๊ฒฐ๊ณผ ์ง๋ํด 2ํ์ ๊ฑธ์ณ 10๋ช์ ์ฑ์ฉํ๋ ๋ฑ ์ ๋ถ์ ์ฒญ๋ ๋ฐ ์ฅ์ ์ธ ์๋ฌด๊ณ ์ฉ์ ๋ํ ๊ธฐ์ค์ ์ถฉ์กฑ์์ผฐ๋ค๋ ํ๊ฐ๋ฅผ ๋ฐ์๋ค. ๋ํ ์ง์ญ๋ด ์ฌํ ์ด๋์์ ์์ ์ ์ธ ์ทจ์์ ๋์์ด ๋  ์ ์๋ ์์ง์ ์ผ์๋ฆฌ ๊ฒฝํ ๋ฐ ์ญ๋์ ์์ ์ ์๋๋ก ์ ๋ผ๋จ๋๊ฐ ์ค์  ์ถ์งํ๊ณ  ์๋ โ์ฒญ๋ ๋ด์ผ๋ก ํ๋ก์ ํธโ์ ์ฐธ์ฌํด 7๋ช์ ์ง์ญ์ธ์ฌ๋ฅผ ์ ๋ฐํ๊ธฐ๋ ํ๋ค. ํนํ ์ ๋จ๊ฐ๋ฐ๊ณต์ฌ์ ์ฑ์ฉ์ ๊ณต์ ์ฑ๊ณผ ํฌ๋ช์ฑ ์ํด ์ ๋ฉด ๋ธ๋ผ์ธ๋ ์ ์ฐจ์ ๋ฐ๋ผ ์งํ๋๋ฉฐ, ํนํ ๋ฉด์ ์ ์ ์ ์ธ๋ถ๋ฉด์ ์์์ผ๋ก ์งํ๋๋ค. ์ฌํด ์ฑ์ฉ์ ์ ๋ฐ๊ธฐ์ 2๋ช์ ์ฑ์ฉํ์ผ๋ฉฐํ๋ฐ๊ธฐ์๋ ์ถ๊ฐ๋ก 5๋ช์ด๋ด์ ๊ท๋ชจ๋ก ์งํ๋  ์์ ์ด๋ค. ํํธ ์ ๋จ๊ฐ๋ฐ๊ณต์ฌ๋ ์ง๋ 2004๋ ์ ๋จ๋๊ฐ ์ค๋ฆฝํ ์ง๋ฐฉ๊ณต๊ธฐ์์ผ๋ก ๋จ์์ ๋์, ๋น๊ฐ๋ ํ์ ๋์, ์ฌ์๊ฒฝ๋ํด์๊ด๊ด๋จ์ง ๊ฐ๋ฐ์ฌ์ ๋ฑ์ ์ํํ๊ณ , ์ฌ์ ์ฃฝ๋ฆผ์ง๊ตฌ ํ์ง๊ฐ๋ฐ์ฌ์๋ ์ถ์งํ  ๊ณํ์ด๋ค.|
|2|๋ชจ๋ธ์์ฝ|์ ๋จ๊ฐ๋ฐ๊ณต์ฌ๋ ์ง๋ 3์ผ ์ธ์ข์ปจ๋ฒค์์ผํฐ์์ ์ด๋ฆฐ 2019๋ ์๋ฐ๊ธฐ ์ง๋ฐฉ๊ณต์ฌยท๊ณต๋จ CEO ๋ฆฌ๋์ญ ํฌ๋ผ์์ ์ผ์๋ฆฌ์ฐฝ์ถ ์ฐ์๊ธฐ๊ด์ ์ ์ ๋ผ ํ์ ์์ ๋ถ์ฅ๊ด ํ์ฐฝ์ ์์ํ๋ค.|
|3|๊ธฐ์ฌ|๊ด์ฃผ์๋ ์ง์ญ ์ ๋ง๊ฐ์๊ธฐ์์ ๋ฐ๊ตดยท์ก์ฑํ๊ธฐ ์ํด ์ด์ํ๊ณ  ์๋ โ100๋ ๋ชํ๊ฐ์๊ธฐ์ ์ก์ฑ์ฌ์โ์ ์ฐธ๊ฐํ  ๊ธฐ์์ ๋ชจ์งํ๋ค. ๋ชํ๊ฐ์๊ธฐ์ ์ก์ฑ์ฌ์์ ์ฑ์ฅ ์ ์ฌ๋ ฅ๊ณผ ๋ฐ์ด๋ ๊ธฐ์ ๋ ฅ์ ๊ฐ์ง ์ง์ญ ์ค์ยท์ค๊ฒฌ๊ธฐ์์ ๋ฐ๊ตดํด ์ง์ญ๊ฒฝ์ ๋ฅผ ๊ฒฌ์ธํ  ๊ธ๋ก๋ฒ ๊ธฐ์์ผ๋ก ์ก์ฑํ๊ธฐ ์ํ ์์ฑ์ผ๋ก, 2014๋ ์์๋ผ ์ฌํด๋ก 6๋์งธ๋ฅผ ๋ง์๋ค. ๋ชจ์ง๋์์ ๊ณต๊ณ ์ผ ํ์ฌ ๋ณธ์ฌ์ ์ฃผ์ฌ์์ฅ์ด ๊ด์ฃผ์ ์์นํ ์ ์กฐ์ ๋ฐ ์ง์์๋น์ค์ฐ์ ๊ธฐ์์ผ๋ก ์ด 30๊ฐ์ฌ๋ค. ์ด๋ฒ ๋ชจ์ง์ ํ์ฌ ์ 3๊ธฐ ๋ชํ๊ฐ์๊ธฐ์ 27๊ฐ์ฌ์ ์ง์ ๊ธฐ๊ฐ(3๋)์ด ๋ง๋ฃ๋จ์ ๋ฐ๋ผ ์ด๋ค ๊ธฐ์์ ์ฌ์ง์  ์ฌ๋ถ์ ํจ๊ป ์ฌ์ง์  ํฌ๊ธฐยทํ๋ฝ ๊ธฐ์ ๊ฒฐ์๋ถ์ ์ฑ์ฐ๊ธฐ ์ํด ์ถ์ง๋๋ค. ์ ์ ์กฐ๊ฑด์ ๋ชํ๊ฐ์๊ธฐ์์ ๋งค์ถ์ก 50์ต์ ์ด์(์ง์์๋น์ค์ฐ์์ 10์ต์ ์ด์)์ด๋ฉด์ ์ต๊ทผ 5๋ ๊ฐ ์ฐํ๊ท  ๋งค์ถ์ก ์ฆ๊ฐ์จ 5% ์ด์์ด๊ฑฐ๋, ์ต๊ทผ 3๋ ๊ฐ ๋งค์ถ์ก ๋๋น R&D ํฌ์ ๋น์จ์ด 1% ์ด์์ธ ๊ธฐ์์ด๋ค. ๋ชํ๊ฐ์๊ธฐ์์ผ๋ก ์ ์ ๋๋ฉด ๊ด์ฃผ์ ์๊ธ ์ง์, ๊ธฐ์์ง๋จ ์ปจ์คํ, ์ฑ์ฅ์ ๋ต ๋ง๋ จ, ํด์ธ๋ง์ผํ ๋ฑ ๊ธฐ์์ค์ฌ ๋ง์ถคํ ์ง์๊ณผ ํจ๊ป ๋ค์ํ ์ฐ๋ ํํ์ ๋ฐ๊ฒ ๋๋ค. ๋ ์ค์์ ๋ถ(์ค์๋ฒค์ฒ๊ธฐ์๋ถ)์ ์ฐ๊ณํ ๊ธฐ์์ฑ์ฅ์ฌ๋ค๋ฆฌ๋ฅผ ํตํด ๋จ๊ณ๋ณ ์ฑ์ฅ์ ๋ต ์ง์๋ ๋ฐ์ ์ ์์ด ๋ชํ๊ฐ์๊ธฐ์ ์ ์  ์ดํ ๊ธ๋ก๋ฒ ๊ธฐ์์ผ๋ก ๋ฐ๋์ํ  ์ ์์ ๊ฒ์ผ๋ก ๊ธฐ๋๋๋ค. ์ ์ฒญ์ 31์ผ๊น์ง ๊ด์ฃผํํฌ๋ธํํฌ๋ก ๋ฐฉ๋ฌธ ์ ์ํ๋ฉด ๋๋ค. ์์ธํ ๋ด์ฉ์ ๊ด์ฃผ์ ํํ์ด์ง(http://www.gwangju.go.kr) ๊ณ ์ยท๊ณต๊ณ ๋์ ์ฐธ๊ณ ํ๊ฑฐ๋ ๊ด์ฃผ์ ๊ธฐ์์ก์ฑ๊ณผ(062-613-3871)๋ก ๋ฌธ์ํ๋ฉด ๋๋ค. ๊ด์ฃผ์๋ ์ ์ฒญ๊ธฐ์์ ๋์์ผ๋ก 1์ฐจ ์๋ฅ์ฌ์ฌ, 2์ฐจ ๋ฐํยทํ์ฅํ๊ฐ๋ฅผ ๊ฑฐ์ณ 8์ ์ ์ ์์ํ์์ ์ต์ข ํ์ ํ  ๊ณํ์ด๋ค.|
|3|๋ชจ๋ธ์์ฝ|๊ด์ฃผ์๋ ์ง์ญ ์ค์ยท์ค๊ฒฌ๊ธฐ์์ ๋ฐ๊ตดํด ์ง์ญ๊ฒฝ์ ๋ฅผ ๊ฒฌ์ธํ  ๊ธ๋ก๋ฒ ๊ธฐ์์ผ๋ก ์ก์ฑํ๊ธฐ ์ํด ์ด์ํ๊ณ  ์๋ '100๋ ๋ชํ๊ฐ์๊ธฐ์ ์ก์ฑ์ฌ์'์ ์ฐธ๊ฐํ  ๊ธฐ์์ ๋ชจ์งํ๋ค.|

## Reference
- [KoBART](https://github.com/SKT-AI/KoBART)
- [KoBART-summarization](https://github.com/seujung/KoBART-summarization)

