# MSRVTT text->video map (base model)

- Data: 1000 paired MSRVTT video-text-standard samples, D=3584.
- Split: 80/20 train/test, random seed 42.

## Test-set performance
- **full_affine_ridge**: R2=0.1882, RMSE=0.0083, mean cosine=0.8682
- **full_linear_ridge**: R2=0.1872, RMSE=0.0083, mean cosine=0.8682
- **diagonal_affine**: R2=0.1069, RMSE=0.0087, mean cosine=0.8540
- **scalar_affine**: R2=-0.9074, RMSE=0.0127, mean cosine=0.6493

## Parameter highlights
- scalar_affine: a=0.6424, b=0.0001
- diagonal_affine: a mean/std=0.3315/0.0691, b norm-like scale std=0.0124
- full_linear_ridge: ||W||_F=5.33
- full_affine_ridge: ||W||_F=5.23, ||b||_2=0.81

Interpretation: if simple affine forms underperform while full linear/affine forms improve R2, cross-dimensional mixing in g(t) is important.