0. 실험 시 requirements.txt에 작성된 패키지를 사용했습니다.

1. (데이터 있으면 건너뛰기) 변수명이 ["connectivity"] 이 되도록 MSN 데이터셋을 생성하고 이를 corr_{subject name}.mat 파일로 저장합니다.
이 파일을 data/SZ_MSN 디렉토리에 배치합니다. MSN 데이터 생성 방법에 대한 지침은 data/SZ_MSN/MSNCconstruction.ipynb를 참조할 수 있습니다.

2. (데이터 있으면 건너뛰기) phenotype.csv 파일을 생성하고 클래스 레이블명은 "DX_GROUP"을 사용합니다.

3. (데이터 있으면 건너뛰기) 실험에 사용할 subject 파일 이름을 subject_IDs.txt 파일에 입력합니다.

4. Parser.py 에서 데이터가 담긴 폴더명을 root_folder에 설정합니다. 본인의 컴퓨터 환경에 맞게 설정해 주세요.

5. 코드를 실행하려면 터미널창에 main.py 을 실행하세요.

6. opt.py 의 값을 기반으로 다양한 하이퍼파라미터를 조정할 수 있습니다.
예를 들어, 다음과 같은 수정을 할 수 있습니다: "python main.py --lr 0.001 --msn_threshold 0.3 --encoder EA --hgc 128"

7. main_rf.py / main_svm.py / main_knn.py / main_dnn.py / main_dnn_batch.py 은 모델 비교 실험을 위한 코드입니다.
main_dnn_batch 와 main_dnn의 차이는 batch processing을 하는지의 차이, batch 사용 권장.