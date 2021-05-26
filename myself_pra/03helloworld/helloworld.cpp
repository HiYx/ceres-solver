//https://ceres-solver.googlesource.com/ceres-solver/+/master/examples/helloworld.cc

//本节将简单描述Ceres的Hello world例程，以便让读者对库的使用步骤快速建立认识。
//在Hello World这个例子中，待优化的函数是 f(x)=10−x 。重载()操作符如下：

//我们建立的lossFunction 是：最小化0.5*(10 - x)^2，使用雅可比矩阵，只用自动differentiation


//在后面的代码中，Ceres通过调用 CostFunctor::operator<T>()来使用这一重载操作符。

//在这个例子中可以令 T = double ，然后仅仅以double类型输出残差值。也可以令 T = Jet 然后输出雅可比矩阵。这一部分在后续教程还有更详细的介绍。
//雅可比矩阵实际上就是对一个含有多个参数的函数f(x) 求一系列一阶偏微分

//用Ceres来实现非线性最小二乘法的优化算法

#include "ceres/ceres.h"
#include "glog/logging.h"
using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solve;
using ceres::Solver;
// A templated cost functor that implements the residual r = 10 -
// x. The method operator() is templated so that we can then use an
// automatic differentiation wrapper around it to generate its
// derivatives.
struct CostFunctor {
  template <typename T>
  bool operator()(const T* const x, T* residual) const {
    residual[0] = 10.0 - x[0];
    return true;
  }
};
int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  // The variable to solve for with its initial value. It will be
  // mutated in place by the solver.
  double x = 5000000;
  const double initial_x = x;
  // Build the problem.
  Problem problem;
  // Set up the only cost function (also known as residual). This uses
  // auto-differentiation to obtain the derivative (jacobian).通过自动差分得到代价函数的导数
  CostFunctor *tempp_yyx = new CostFunctor;
  CostFunction* cost_function =
      new AutoDiffCostFunction<CostFunctor, 1, 1>(tempp_yyx);
  problem.AddResidualBlock(cost_function, nullptr, &x);//传入代价函数与初值
  // Run the solver!
  Solver::Options options;
  options.minimizer_progress_to_stdout = true;
  Solver::Summary summary;
  Solve(options, &problem, &summary);
  std::cout << summary.BriefReport() << "\n";
  std::cout << "x : " << initial_x << " -> " << x << "\n";
  return 0;
}
