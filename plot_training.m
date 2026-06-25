%% plot_training.m
% MATLAB脚本 - 读取训练历史数据并绘制高质量曲线图
% 使用方法: 在MATLAB中运行此脚本, 确保 training_history.mat 在同一目录下
%
% 图片来源: 实验数据由Python训练代码生成, MATLAB脚本自行编写

clear; clc; close all;

%% 1. 加载数据
data = load('training_history.mat');
epoch = data.epoch;
train_loss = data.train_loss;
val_loss = data.val_loss;
train_acc = data.train_acc;
val_acc = data.val_acc;

%% 2. 绘图设置
set(0, 'DefaultAxesFontSize', 12);
set(0, 'DefaultTextFontSize', 12);
set(0, 'DefaultLineLineWidth', 2);

%% 3. 绘制损失曲线和准确率曲线
figure('Position', [100, 100, 1200, 450]);

% ---- 左图: 损失曲线 ----
subplot(1, 2, 1);
plot(epoch, train_loss, 'b-', 'LineWidth', 2, 'DisplayName', 'Training Loss');
hold on;
plot(epoch, val_loss, 'r-', 'LineWidth', 2, 'DisplayName', 'Validation Loss');
xlabel('Epoch', 'FontSize', 13);
ylabel('Loss', 'FontSize', 13);
title('Training and Validation Loss', 'FontSize', 14, 'FontWeight', 'bold');
legend('Location', 'northeast', 'FontSize', 11);
grid on;
grid minor;
hold off;

% ---- 右图: 准确率曲线 ----
subplot(1, 2, 2);
plot(epoch, train_acc, 'b-', 'LineWidth', 2, 'DisplayName', 'Training Accuracy');
hold on;
plot(epoch, val_acc, 'r-', 'LineWidth', 2, 'DisplayName', 'Validation Accuracy');
xlabel('Epoch', 'FontSize', 13);
ylabel('Accuracy (%)', 'FontSize', 13);
title('Training and Validation Accuracy', 'FontSize', 14, 'FontWeight', 'bold');
legend('Location', 'southeast', 'FontSize', 11);
grid on;
grid minor;

% 标记最佳验证准确率
[best_acc, best_idx] = max(val_acc);
hold on;
plot(epoch(best_idx), best_acc, 'go', 'MarkerSize', 10, ...
     'MarkerFaceColor', 'g', 'DisplayName', sprintf('Best: %.2f%%', best_acc));
hold off;

%% 4. 保存图片
sgtitle('ResNet-50 迁移学习 - 24种水果分类训练曲线', 'FontSize', 16, 'FontWeight', 'bold');
saveas(gcf, 'training_history_matlab.png');
fprintf('图片已保存: training_history_matlab.png\n');
fprintf('最佳验证准确率: %.2f%% (Epoch %d)\n', best_acc, epoch(best_idx));

%% 5. 打印数值表格
fprintf('\n===== 训练日志 =====\n');
fprintf('%-8s %-14s %-14s %-14s %-14s\n', 'Epoch', 'Train Loss', 'Val Loss', 'Train Acc%', 'Val Acc%');
fprintf('%s\n', repmat('-', 1, 65));
for i = 1:length(epoch)
    fprintf('%-8d %-14.4f %-14.4f %-14.2f %-14.2f\n', ...
        epoch(i), train_loss(i), val_loss(i), train_acc(i), val_acc(i));
end
