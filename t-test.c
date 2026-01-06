#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>


// Box-Muller Transform.
double random_normal(double mu, double sigma) {
    static double U;
    static double V;
    static int phase = 0;
    double Z;

    if (phase == 0) {
        U = (rand() + 1.0) / (RAND_MAX + 1.0);
        V = (rand() + 1.0) / (RAND_MAX + 1.0);
        Z = sqrt(-2.0 * log(U)) * sin(2.0 * M_PI * V);
    } else {
        Z = sqrt(-2.0 * log(U)) * cos(2.0 * M_PI * V);
    }
    phase = 1 - phase;

    return Z * sigma + mu;
}


double incbeta(double a, double b, double x) {
    static const double epsilon = 1.0e-30;
    static const double condition = 1.0e-8;

    if (x < 0.0 || x > 1.0) return INFINITY;
    if (x > (a + 1.0) / (a + b + 2.0)) return 1.0 - incbeta(b, a, 1.0 - x);

    double f = 1.0;
    double c = 1.0;
    double d = 0.0;
    const double lbeta_ab = lgamma(a) + lgamma(b) - lgamma(a + b);
    const double front = exp(a * log(x) + b * log(1.0 - x) - lbeta_ab) / a;

    for (int i = 0; i <= 200; ++i) {
        int m = i / 2;
        double numerator;

        if (i == 0) numerator = 1.0;
        else if (i % 2 == 0) numerator = (m * (b - m) * x) / ((a + 2.0 * m - 1.0) * (a + 2.0 * m));
        else numerator = - ((a + m) * (a + b + m) * x) / ((a + 2.0 * m) * (a + 2.0 * m + 1));

        d = 1.0 + numerator * d;
        if (fabs(d) < epsilon) d = epsilon;
        d = 1.0 / d;

        c = 1.0 + numerator / c;
        if (fabs(c) < epsilon) c = epsilon;

        double cd = c * d;
        f = f * cd;
        if (fabs(1.0 - cd) < condition) return front * (f - 1.0);
    }
    return INFINITY;
}


double cdf_student_t(double t, double df) {
    /*
    double x = df / (t * t + df);
    double p = 0.5 * incbeta(df / 2.0, 0.5, x);
    return (t > 0) ? (1.0 - p) : p;
    */
    if (df <= 0.0) return NAN;
    if (t == 0.0) return 0.5;
    double cache = sqrt(t * t + df);
    return incbeta(df / 2.0, df / 2.0, (t + cache) / (2.0 * cache));
}


double ttest_independent(double *group1, double *group2, int n1, int n2, int equal_var) {
    double sum1 = 0.0;
    double sum2 = 0.0;
    for (int i = 0; i < n1; ++i) sum1 = sum1 + group1[i];
    for (int i = 0; i < n2; ++i) sum2 = sum2 + group2[i];
    double mean1 = sum1 / n1;
    double mean2 = sum2 / n2;

    double var1 = 0.0;
    double var2 = 0.0;
    for (int i = 0; i < n1; ++i) var1 = var1 + (group1[i] - mean1) * (group1[i] - mean1);
    for (int i = 0; i < n2; ++i) var2 = var2 + (group2[i] - mean2) * (group2[i] - mean2);
    var1 = var1 / (n1 - 1);
    var2 = var2 / (n2 - 1);

    double t;
    double df;

    if (equal_var) {
        df = n1 + n2 - 2.0;
        double pooled_var = ((n1 - 1.0) * var1 + (n2 - 1.0) * var2) / df;
        t = (mean1 - mean2) / sqrt(pooled_var * (1.0 / n1 + 1.0 / n2));
    } else {
        double satterthwaite1 = var1 / n1;
        double satterthwaite2 = var2 / n2;
        t = (mean1 - mean2) / sqrt(satterthwaite1 + satterthwaite2);
        double numerator = (satterthwaite1 + satterthwaite2) * (satterthwaite1 + satterthwaite2);
        double denominator = (satterthwaite1 * satterthwaite1) / (n1 - 1.0) + (satterthwaite2 * satterthwaite2) / (n2 - 1.0);
        df = numerator / denominator;
    }

    double cdf = cdf_student_t(t, df);
    return 2.0 * (cdf > 0.5 ? 1.0 - cdf : cdf);
}


double ttest_paired(double *state1, double *state2, int n) {
    if (n < 2) return NAN;

    double sum_diff = 0.0;
    double sum_square_diff = 0.0;
    for (int i = 0; i < n; ++i) {
        double diff = state1[i] - state2[i];
        sum_diff = sum_diff + diff;
        sum_square_diff = sum_square_diff + diff * diff;
    }

    double mean_diff = sum_diff / n;
    double var_diff = (sum_square_diff - (sum_diff * sum_diff / n)) / (n - 1);
    double std_error = sqrt(var_diff / n);

    double t = mean_diff / std_error;
    double cdf = cdf_student_t(t, n - 1.0);
    double p = 2.0 * (cdf > 0.5 ? 1.0 - cdf : cdf);
    return p;
}



int main(int argc, char *argv[], char *envs[]) {
    double p;
    srand((unsigned int)time(NULL));

    int n1 = 6;
    int n2 = 4;
    double mu1 = 75.0;
    double mu2 = 82.0;
    double sigma1 = 8.0;
    double sigma2 = 8.0;

    double category1[n1];
    double category2[n2];

    for (int i = 0; i < n1; ++i) category1[i] = random_normal(mu1, sigma1);
    for (int i = 0; i < n2; ++i) category2[i] = random_normal(mu2, sigma2);
    p = ttest_independent(category1, category2, n1, n2, sigma1 == sigma2);
    printf("Independent Samples T-test:\x1b[32m p = %.12lf\x1b[0m\n", p);

    double before[] = {22, 20, 19, 24, 25, 25, 28, 22, 30, 27, 24, 18, 16, 19, 19, 28, 24, 25, 25, 23};
    double after[] = {24, 22, 19, 22, 28, 26, 28, 24, 30, 29, 25, 20, 17, 18, 18, 28, 26, 27, 27, 24};
    p = ttest_paired(before, after, sizeof(after) / sizeof(after[0]));
    printf("Paired Sample T-test:\x1b[32m p = %.12lf\x1b[0m\n", p);

    return 0;
}