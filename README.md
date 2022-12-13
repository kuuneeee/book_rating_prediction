# 📚 Book Rating Prediction

***description*** :

> **GOAL** : 책과 관련된 정보와 소비자의 정보, 그리고 소비자가 실제로 부여한 평점, 총 3가지의 데이터 셋(users.csv, books.csv, train_ratings.csv)을 활용하여 각 사용자가 주어진 책에 대해 얼마나 평점을 부여할지에 대해 예측

**input**

- `train_ratings.csv` : 각 사용자가 책에 대해 평점을 매긴 내역

![image](https://user-images.githubusercontent.com/79534756/207330407-95496db8-473e-4e66-9781-8c10562fed69.png)

- `users.csv` : 사용자에 대한 정보

![image](https://user-images.githubusercontent.com/79534756/207330772-82450b82-fa9d-4f2d-9c87-c6c5cb7366fe.png)

- `books.csv` : 책에 대한 정보

![image](https://user-images.githubusercontent.com/79534756/207330974-f8b871b1-5828-4fce-b40f-94475c37e29a.png)

- `Image/` : 책 이미지

![image](https://user-images.githubusercontent.com/79534756/207331196-770b2b43-fc8b-4c78-8976-333381e02e8b.png)



- ***Metric*** : 

	- Book Rating Prediction은 사용자가 그동안 읽은 책에 부여한 평점 데이터를 사용해서 새로운 책을 추천했을 때 어느 정도의 평점을 부여할지 예측하는 문제
	
	- 즉, Regression(회귀) 문제로 볼 수 있으며, 평점 예측에서 자주 사용되는 지표 중 하나인 RMSE (Root Mean Square Error)를 사용한다.



## 📁프로젝트 구조

```
├─models
├─src
	├─models
	├─ensembles
	├─data
├─submit
├─main.py
├─ensemble.py
├─requirements.txt
```



