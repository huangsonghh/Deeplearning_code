clear;
clc;
close all
addpath('tSVD','proxFunctions','solvers','twist','Image_data','tools');
addpath('ClusteringMeasure', 'LRR', 'Nuclear_norm_l21_Algorithm', 'unlocbox');

dataset = 'bbc4view'; % 'bbcsport_2view'
%dataset = 'bbcsport_2view'; % 'bbcsport_2view'
load(['./',dataset,'.mat'])

result = [];

cls_num = length(unique(gt));
inXCell = X; %用inXCell是为了提示你X用的是元胞数组的格式吧
clear X
N = size(inXCell{1},2); % sample number
K = numel(inXCell); % view number     numel函数用于计算数组中元素的个数

% Normalize the data
for k=1:K
    inXCell{k}=NormalizeData(inXCell{k});
end

alphaGridSearchSet = [5];  %alpha的取值集合，alpha是噪声前的系数0.05,0.1,0.5,1,5,10
dGridSearchSet = [3];  %很好奇这个集合是如何设定的？1,1.5,2,3,4

for mm=1:numel(alphaGridSearchSet)
    for nn=1:numel(dGridSearchSet)
        alpha = alphaGridSearchSet(mm)
        beta = 5e-8  %beta是k-means目标函数前的系数
        d = ceil(dGridSearchSet(nn)*cls_num)  %d相当于论文中的m，这样看来它确实要远小于特征数和样本数
        
        % Initialize Matrix U, Z, H and G
        for k=1:K
            W{k} = zeros(size(inXCell{k},1),d);
            H{k} = ones(d,N); %为什么表示矩阵一开始要令成全是1的矩阵？
            U{k} = zeros(d,N);  %这里U代表的是论文里的辅助变量Z
            E{k} = zeros(size(inXCell{k},1),N);
            Y{k} = zeros(size(inXCell{k},1),N);
            M{k} = zeros(d,N);
            D{k} = ones(1, N); %注意D令成了一个行向量
            F{k} = ones(d,cls_num);
        end
        sig = zeros(min(d,N),1); % for DC 
        % initialize the cluster indicator matrix G (c*N)
        [G0,label_in] = initial_input(cls_num,N); %注意这里得到的G0是n*k的矩阵，label_in中存储的是每个元素记录着对应样本的中心下标
        G = G0'; %G得到的才是一个k*n的矩阵
        
        H_tensor = 0;
        M_tensor = 0;
        m = zeros(d*N*K,1); %记录的是M的向量化
        h = zeros(d*N*K,1);
        dim1 = d;dim2 = N;dim3 = K;  %记录的是H_tensor的大小
        myNorm = 'tSVD_1';
        sX = [d, N, K];
        
        %% set Default   （这是默认设置吗？）
        parOP = false;
        ABSTOL = 1e-6;
        RELTOL = 1e-4;
        Isconverg = 0;
        epson = 1e-4;
        iter = 0;
        mu = 1e-8; max_mu = 10e6; pho = 2;
        gamma=1e-3;     %gamma parameter in the rank approximation
        
        tic;
        %% main procedure
        while(Isconverg == 0)
            fprintf('----processing iter %d--------\n', iter+1);
            %1 update W
            % with orthometric constraint
            disp('Processing W Subproblem...')
            for k=1:K
                W_svd = H{k}*(inXCell{k}-E{k}+Y{k}/mu)'; %这样的话最后要转置，那为什么不直接(inXCell{k}-E{k}+Y{k}/mu)*H{k}'呢？，难道那样更快？
                [W_U,W_S,W_V] = svd(W_svd,'econ');
                W_temp = W_U * W_V';
                W{k} = W_temp';
            end
            clear W_svd W_U W_V W_S   %为了少占用一些内存吧
            disp('W Subproblem Done!')
            
            %2 update H
            disp('Processing H Subproblem...')
            for k=1:K
                H_A = mu*(W{k}'*W{k}+eye(d));
                H_B_vec = 2*beta*D{k}.^2; %注意H_B_vec是一个行向量
                H_C = mu*W{k}'*(inXCell{k}-E{k}+Y{k}/mu)+mu*(U{k}+M{k}/mu) + 2*beta*F{k}*G.*repmat(D{k},[d,1]); %repmat(D{k},[d,1]）与repmat(D{k},d,1）效果一样,注意F*G后的大小与repmat作用D后一致
                for i=1:N
                    H{k}(:,i) = (H_A+H_B_vec(i)*eye(d))\H_C(:,i); %注意：这里是左除"\"，例如A\B=inv(A)*B
                end
            end
            clear H_A H_B_vec H_C
            disp('H Subproblem Done!')
            
            %3 update D
            for k=1:K
                Hi2 = sqrt(sum(H{k}.*H{k}, 1) + eps); %这里加一个eps是为了防止0开根号吗
                D{k} = 1./Hi2;
            end
            clear Hi2
            
            %4 update F
            for k=1:K
                F_M = G*G'+diag(eps*ones(1,cls_num)); %后面加上的这一部分是为了保证它可逆吗？
                F_N = H{k}.*repmat(D{k},[d,1])*G';
                F{k} = F_N*pinv(F_M);
            end
            clear F_M F_N
            
            %5 update G
            disp('Processing G Subproblem...')
            for i = 1:N
                dVec = zeros(K, 1);
                for k = 1:K
                    xVec{k} = H{k}(:,i)*D{k}(i);
                    dVec(k, 1) = 1;
                end
                G0(i,:) = searchBestIndicator(dVec, xVec, F);
            end
            G = G0';
            disp('G Subproblem Done!')
            
            %6 update E
            disp('Processing E Subproblem...')
            for k=1:K
                E_svt = inXCell{k}-W{k}*H{k}+Y{k}/mu;
                [E{k}] = solve_l1l2(E_svt,alpha/mu);
            end
            clear E_svt
            disp('E Subproblem Done!')
            
            %7 update U
            % H_tensor = cat(3, H{:,:});
            % M_tensor = cat(3, M{:,:});
            % h = H_tensor(:);
            % m = M_tensor(:);
            % 
            % %twist-version（旋转版本）
            % disp('Processing Tensor U Subproblem...')
            % [u, objV] = wshrinkObj(h - 1/mu*m,1/mu,sX,1,3);
            % U_tensor = reshape(u, sX);
            % disp('Tensor U Subproblem Done!')

            %%非凸版本
            disp('Processing Tensor U Subproblem...')
            H_tensor = cat(3, H{:,:});
            M_tensor = cat(3, M{:,:});
            h = H_tensor(:);
            m = M_tensor(:);
            H_tensor_f = fft(H_tensor, [], 3);
            M_tensor_f = fft(M_tensor, [], 3);
            for k=1:K
                H_f{k} = H_tensor_f(:,:,k);
                M_f{k} = M_tensor_f(:,:,k);
                [U{k},sig] = DC(H_f{k}-1/mu.*M_f{k},mu/(2*K),sig,gamma);%第二个参数可能有问题需要在检查一下
            end
            U_tensor_f = cat(3, U{:,:});
            U_tensor = ifft(U_tensor_f, [], 3);
            u = U_tensor(:);
            disp('Tensor U Subproblem Done!')

           
            
            %8 update lagrange multiplier Y and M
            for k=1:K
                Y{k} = Y{k} + mu*(inXCell{k}-W{k}*H{k}-E{k});
            end
            m = m + mu*(u - h); %转换为向量来计算是为了加速运算吗
            
            % coverge condition
            Isconverg = 1;
            for k=1:K
                U{k} = U_tensor(:,:,k);
                if (norm(U{k}-H{k},inf)>epson)
                    history.norm_U_H = norm(U{k}-H{k},inf);
                    fprintf('norm_U_H %7.10f    \n', history.norm_U_H);
                    Isconverg = 0;
                end
                temp_E = inXCell{k} - W{k}*H{k};
                if (norm(temp_E-E{k},inf)>epson)
                    history.norm_E = norm(temp_E-E{k},inf);
                    fprintf('norm_E %7.10f    \n', history.norm_E);
                    Isconverg = 0;
                end
            end
            
            if (iter>60)
                Isconverg  = 1;
            end
            iter = iter + 1;
            mu = min(mu*pho, max_mu);
        end
        toc;
        
        temp = zeros(N,1);
        for i=1:N
            temp(i) = find(G(:,i)==1);
        end
        Cluster_result = temp;
        
        % external measure
        [A nmi avgent] = compute_nmi(gt,Cluster_result);
        ACC = Accuracy(Cluster_result,double(gt));
        [f,p,r] = compute_f(gt,Cluster_result);
        [AR,RI,MI,HI]=RandIndex(gt,Cluster_result);
        
        fprintf(['JSTC: alpha = %f, d = %d, NMI=%.3f, ACC=%.3f\n'],  alpha, d, nmi, ACC)
        result = [result;[alpha,d, nmi, ACC, AR, f, p, r, RI]];
    end
end
save(['./result_JSTC_',dataset,'.mat'], 'gt','result','alphaGridSearchSet', 'dGridSearchSet')


