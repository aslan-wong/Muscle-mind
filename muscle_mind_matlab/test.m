
a = my_test(2,2);

function out = my_test(x,b,varargin)
    p = inputParser;            % 函数的输入解析器
    addParameter(p,'k',1);      % 设置变量名和默认参数
    parse(p,varargin{:});       % 对输入变量进行解析，如果检测到前面的变量被赋值，则更新变量取值
    out = p.Results.k*x + b;    % 在这里定义你自己的函数计算公式
end
