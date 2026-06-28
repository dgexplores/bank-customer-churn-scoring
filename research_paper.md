Analyzing customer churn in banking: A data mining framework
Authors: Aishwarya Saxena, Anushi Singh, Govindaraj M.
Multidiscip. Sci. J. (2023) 5:e2023ss0310
Published Online: August 29, 2023
DOI: https://doi.org/10.31893/multiscience.2023ss0310

================================================================
ABSTRACT
================================================================
Customer churn, the loss of customers to a business, is a significant challenge in
the banking industry. Retaining existing customers is crucial for banks to maintain
profitability and sustain growth. This paper focuses on analyzing customer churn in
the banking sector. The study utilizes data mining and predictive analytics
techniques to analyse customer behaviour, identify churn patterns, and develop
predictive models. This research uses a data mining technique called Gaussian
mixture model clustering-based adaptive support vector machine (GMM-ASVM) to
forecast customer loss in the banking industry. By analyzing consumer competency
and loyalty to the banking industry using GMM, this study predicts customer
behaviour using a clustering approach. An accuracy of 98% was attained while
classifying the clustering results using ASVM.

Keywords: customer churn, GMM-ASVM, banking industry

================================================================
TABLE 1: DATASET DESCRIPTION
================================================================
Attribute            | Description
---------------------|----------------------------------------------
Customer ID          | ID of customer
Row Number           | Number of customers
Geography            | Location of customer
Age                  | Age of Customer
Gender               | Customer gender
Surname              | Customer name
Credit Score         | Score of credit card usage
No. of. Products     | No of products used by customer
Tenure               | The period of having the account in months
Estimated Salary     | Estimated salary of the customer
Churn                | Indicates customer leaved or not

Note: Dataset source - Kaggle, 10,000 bank customer records, 14 variables total.

================================================================
METHODOLOGY OVERVIEW (Figure 1)
================================================================
Input data -> Data preprocessing -> Clustering using Gaussian Mixture Model
-> SVM prediction -> Performance Evaluation

Data pre-processing steps:
1. Irrelevancy removal: Row Number, Customer Id, Surname, and Geography were
   manually removed as they don't affect churn prediction.
2. Transformation: Data converted/formatted (handling null values, duplication,
   indexing issues, file format incompatibility).

Model building:
Step 1 - Clustering using Gaussian Mixture Model (GMM)
Step 2 - Prediction using Adaptive SVM (ASVM), using RBF kernel and GridSearchCV
         for hyperparameter tuning (C, gamma)

================================================================
TABLE 2: VALUES OF ACCURACY (%)
================================================================
No. of Sample | CNN (De Caigny 2020) | DT-MR (Rouhani & Mohammadi 2022) | ANN (Yahaya 2021) | GMM-ASVM (Proposed)
1             | 82.8                  | 83.62                             | 85.75              | 89.92
2             | 81.5                  | 82.25                             | 86.75              | 92.15
3             | 83.5                  | 84.65                             | 87.25              | 94.75
4             | 84.2                  | 86.25                             | 89.35              | 96.32
5             | 85.3                  | 87.35                             | 90.15              | 98

================================================================
TABLE 3: VALUES OF PRECISION (%)
================================================================
No. of Sample | CNN (De Caigny 2020) | DT-MR (Rouhani & Mohammadi 2022) | ANN (Yahaya 2021) | GMM-ASVM (Proposed)
1             | 83.75                 | 86.25                             | 84.15              | 90.35
2             | 85.5                  | 88.75                             | 85.12              | 92.15
3             | 87.15                 | 89.25                             | 90.15              | 94.51
4             | 88.6                  | 90.15                             | 91.85              | 95.41
5             | 90.75                 | 91.62                             | 92.85              | 97.22

================================================================
TABLE 4: VALUES OF RECALL (%)
================================================================
No. of Sample | CNN (De Caigny 2020) | DT-MR (Rouhani & Mohammadi 2022) | ANN (Yahaya 2021) | GMM-ASVM (Proposed)
1             | 80.15                 | 85.25                             | 84.75              | 88.25
2             | 83.12                 | 82.25                             | 85.15              | 89.14
3             | 85.25                 | 83.85                             | 86.25              | 91.22
4             | 84.75                 | 85.15                             | 89.75              | 93.17
5             | 86.88                 | 86.1                              | 90.15              | 94.24

================================================================
TABLE 5: VALUES OF COMPUTATIONAL TIME (seconds)
================================================================
Method                                  | Computational Time (s)
-----------------------------------------|------------------------
CNN (De Caigny et al 2020)               | 1.5
DT-MR (Rouhani and Mohammadi 2022)       | 1.2
ANN (Yahaya et al 2021)                 | 1
GMM-ASVM [Proposed]                      | 0.9

================================================================
KEY EQUATIONS
================================================================
(1) Gaussian Mixture Model:
    p(Y|lambda) = sum_j( omega_j * h(Y|mu_j, Sigma_j) ), j = 1,...,N

(2) Component density function (D-variate Gaussian):
    h(Y|mu_j, Sigma_j) = 1 / ((2*pi)^(E/2) * |Sigma_j|^(1/2))
                          * exp{ -1/2 * (y - mu_j)' * Sigma_j^-1 * (y - mu_j) }
    Constraint: sum_j(omega_j) = 1, j = 1,...,N

(3) SVM classification function:
    f(y) = x*U*y + c
    (x = weight, U = carriage, y = input, c = bias)

(4) Optimization objective:
    Min_(x,c) (1/2)||x||^2

(5) Constraint:
    z_j * (x.y + c) >= 1, for any j = 1,...,n

(6) RBF Kernel (form 1):
    L(y, y') = exp( -||y - y'||^2 / (2*sigma^2) )

(7) RBF Kernel (form 2):
    L(y, y') = exp( -gamma * ||y - y'||^2 )

================================================================
CONCLUSION (Summary)
================================================================
The study uses a Gaussian Mixture Model clustering-based Adaptive Support Vector
Machine (GMM-ASVM) to predict customer churn in banking. GMM clusters customers
based on competency/loyalty, then ASVM classifies the clustering results with 98%
accuracy. The proposed GMM-ASVM outperformed CNN, DT-MR, and ANN baselines across
accuracy, precision, recall, and computational time.

Ethical considerations: Not applicable.
Declaration of interest: No conflicts of interest declared.
Funding: This research did not receive any financial support.

================================================================
REFERENCES
================================================================
Amuda KA, Adeyemo AB (2019) Customers' churn prediction in financial institutions
using artificial neural network. arXiv preprint arXiv:1912.11346.

Dalmia H, Nikil CV, Kumar S (2020) Churning of bank customers using supervised
learning. In Innovations in Electronics and Communication Engineering: Proceedings
of the 8th ICIECE 2019, pp 681-691. Springer Singapore.

De Caigny A, Coussement K, De Bock KW, Lessmann S (2020) Incorporating textual
information in customer churn prediction models based on a convolutional neural
network. International Journal of Forecasting 36:1563-1578.

de Lima Lemos RA, Silva TC, Tabak BM (2022) Propension to customer churn in a
financial institution: A machine learning approach. Neural Computing and
Applications 34:11751-11768.

Gholamiangonabadi D, Nakhodchi S, Jalalimanesh A, Shahi A (2019) Customer churn
prediction using a meta-classifier approach; A case study of the Iranian banking
industry. In Proceedings of the International Conference on Industrial Engineering
and Operations Management, pp 364-375.

Imron MA, Prasetyo B (2020) Improving algorithm accuracy k-nearest neighbor using
z-score normalization and particle swarm optimization to predict customer churn.
Journal of Soft Computing Exploration 1:56-62.

Kaur I, Kaur J (2020) Customer churn analysis and prediction in the banking
industry using machine learning. In 2020 Sixth International Conference on
Parallel, Distributed and Grid Computing (PDGC), pp 434-437. IEEE.

Muneer A, Ali RF, Alghamdi A, Taib SM, Almaghthaw A, Ghaleb EAA (2022) Predicting
customers churning in the banking industry: A machine learning approach.
Indonesian Journal of Electrical Engineering and Computer Science 26:539-549.

Rouhani S, Mohammadi A (2022) A Novel Hybrid Forecasting Approach for Customers
Churn in Banking Industry. Journal of Information & Knowledge Management 2250089.

Satria WA, Fitri I, Ningsih S (2020) Prediction of Customer Churn in the Banking
Industry Using Artificial Neural Networks. Jurnal Mantik 4:936-943.

Sjarif N, Rusydi M, Yusof M, Hooi D, Wong T, Yaakob S, Ibrahim R, Osman M (2019) A
customer Churn prediction using Pearson correlation function and K nearest
neighbor algorithm for the telecommunication industry. Int. J. Advance Soft Compu.
Appl. 11.

Sun Y (2021) Case-based models of the relationship between consumer resistance to
innovation and customer churn. Journal of Retailing and Consumer Services
61:102530.

Tao D, Yang P, Feng H (2020) Utilization of text mining as a big data analysis
tool for food science and nutrition. Comprehensive reviews in food science and
food safety 19:875-894.

Wu Z, Li Z (2021) Customer churn prediction for commercial banks using
customer-value-weighted machine learning models. Journal of Credit Risk 17.

Yahaya R, Abisoye OA, Bashir SA (2021) An enhanced bank customers churn
prediction model using a hybrid genetic algorithm and k-means filter and
artificial neural network. In 2020 IEEE 2nd International Conference on Cyberspac
(CYBER NIGERIA), pp 52-58. IEEE.
