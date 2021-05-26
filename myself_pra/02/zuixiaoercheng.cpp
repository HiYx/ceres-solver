#include <iostream>
#include <fstream>
#include "ceres/ceres.h"

using namespace ceres;

 //定义CostFunctor(i.e.函数 f ),以 f(x)=10-x 为例
//使用自动求导(当使用自动求导时, operator() 是一个模板函数)
struct AutoDiffCostFunctor {
    template <typename T>
    bool operator()(const T* const x, T* residual) const { //修饰大括号的const必须有
        residual[0] = T(10.0) - x[0];
        //10.0必须强制转换为T类型,否则会编译器会报无法实现运算操作的类型
        return true;
    }
};

//使用数值求导
struct NumericDiffCostFunctor {
    bool operator()(const double* const x, double* residual) const {
        residual[0] = 10.0 - x[0];
        return true;
    }
};

//使用函数导数的解析式求导
class QuadraticCostFunction: public ceres::SizedCostFunction<1,1> {
public:
    virtual ~QuadraticCostFunction() {}
    virtual bool Evaluate(double const* const* parameters,//二维参数块指针
    double* residuals,
    //一维残差指针
    double** jacobians) const { //二维jacobian矩阵元素指针,与parameters对应
        residuals[0] = 10.0 - parameters[0][0];
        if(jacobians && jacobians[0]) {
            jacobians[0][0] = -1;
        }
        return true;
    }
}; 
//雅可比矩阵是多元函数一阶偏导数以一定方式排列成的矩阵，体现了一个可微方程与给出点的最优线性逼近。

//二阶导数构成的矩阵为Hessian矩阵(海森矩阵)


int main(int argc, char** argv)
{
   const double initial_x = 0.2;
    double x = initial_x;Problem problem;
    Solver::Options options;
    //control whether the log is output to STDOUT
    options.minimizer_progress_to_stdout = false;
    Solver::Summary summary;
    //AutoDiff
    CostFunction* CostFunction =
        new AutoDiffCostFunction<AutoDiffCostFunctor,1,1>(
            new AutoDiffCostFunctor);
    problem.AddResidualBlock(CostFunction,NULL,&x);
    clock_t solve_start = clock();
    //'summary' must be non NULL
    Solve(options, &problem, &summary);
    clock_t solve_end = clock();
    std::cout << "AutoDiff time : " << solve_end-solve_start << std::endl;
    std::cout << "x : " << initial_x << " -> " << x << std::endl;
    //NumericDiff
    CostFunction =
        new NumericDiffCostFunction<NumericDiffCostFunctor,CENTRAL,1,1>(
            new NumericDiffCostFunctor);
    x = 0.2;
    problem.AddResidualBlock(CostFunction,NULL,&x);
    solve_start = clock();
    Solve(options, &problem, &summary);
    solve_end = clock();
    std::cout << "NumercDiff time : " << solve_end-solve_start << std::endl;
    std::cout << "x : " << initial_x << " -> " << x << std::endl;
    //AnalyticDiff
    //QuadraticCostFunction继承自SizedCostFunction,而SizedCostFunction继承自
    //CostFunction,因此此语句与上述两种对CostFunction的赋值操作略有不同
    CostFunction = new QuadraticCostFunction;
    x = 0.2;
    problem.AddResidualBlock(CostFunction,NULL,&x);
    solve_start = clock();
    Solve(options, &problem, &summary);
    solve_end = clock();
    std::cout << "AnalyticDiff time : " << solve_end-solve_start << std::endl;
    std::cout << "x : " << initial_x << " -> " << x << std::endl; 
    return 0;
};