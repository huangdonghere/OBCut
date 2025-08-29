%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                      %
% This is the source code for the OBCut algorithm, which is proposed   %
% in the following paper:                                              %
%                                                                      %
% Si-Guo Fang, Dong Huang, Chang-Dong Wang, Jian-Huang Lai.            %
% One-step Bipartite Graph Cut: A Normalized Formulation and Its       %
% Application to Scalable Subspace Clustering.                         %
% Neural Networks, accepted, 2026.                                     %
%                                                                      %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function preY = OBCut(X,K,M,lambda)
%% The non-paralleled version
%% Input:
%  X: Data matrix
%  K: Number of clusters
%  M: Number of anchors
%  lambda: Trade-off parameter
%% Output:
%  preY: Clustering result

[N, ~] = size(X); % The number of samples

%Normalization
X = ( X-repmat(min(X),N,1) ) ./repmat( max(X)-min(X),N,1);
X( isnan(X) ) = 1;

%Initialize A
rng(123,'twister')
[~, A] = litekmeans(X,M,'MaxIter',20,'Replicates',2);
A = A';

X = X';

%Initialize B
B = updateB(X,A,rand(N,M),zeros(N,K),zeros(M,K),0,N,K);
opts.isnor = 1;
B = full(my_Bi2KNN(B,opts));

%Initialize Y
[H,Y] = InitY(X,B,K,N);

IterMax = 30;
IterY = 10;

res = zeros(IterMax + 1,1);
res(1) = obj_value(X,A,B,Y,H,lambda);

for i = 1:IterMax
    % Update Y
    tempY = Y;
    for t = 1:IterY
        Y = updateY_my(Y,B * H,N,K);
        if isequal(tempY,Y)
            break;
        end
        tempY = Y;
    end
    
    % Update H
    [U,~,V] = svd(B' * Y * (Y' * Y + eps * eye(K))^-0.5,'econ');
    H = U * V';
    
    % Update B
    B = updateB(X,A,B,Y,H,lambda,N,K);
        
    % Update A
    A = X * B * pinv(B' * B);
    
    res(i+1) = obj_value(X,A,B,Y,H,lambda);
    if abs(res(i+1) - res(i)) < 1e-5 || abs(res(i+1) - res(i)) / abs(res(i)) <1e-5
        break
    end
end

[~,preY] = max(Y,[],2);

end


function res = obj_value(X,A,B,Y,H,lambda)
res = 0;
res = res + norm(X - A * B','fro')^2;    
res = res - lambda * trace((Y' * Y)^-0.5 * Y' * B * H);
end


function [H,Y] = InitY(X,B,K,N)

% Init Y
rng(123,'twister');
labels = litekmeans(X',K,'MaxIter',20,'Replicates',2);
Y = zeros(N,K);
for i = 1:N
    Y(i,labels(i)) = 1;
end

% Init H
[U,~,V] = svd(B' * Y * (Y' * Y + eps * eye(K))^-0.5,'econ');
H = U * V';

end


function Z = my_Bi2KNN(D,opts)

if (~exist('opts','var'))
   opts = [];
end

r = 5;
if isfield(opts,'r')
    r = opts.r;
end

isnor = 0;
if isfield(opts,'isnor')
    isnor = opts.isnor;
end



[nSmp,p] = size(D);

dump = zeros(nSmp,r);
idx = dump;
for i = 1:r
    [dump(:,i),idx(:,i)] = max(D,[],2);
    temp = (idx(:,i)-1)*nSmp+[1:nSmp]';
    D(temp) = -1e100;
end

%dump = exp(-dump/(2*sigma^2));
%dump = dump./repmat(sum(dump,2),1,size(dump,2));

if isnor
    sumD = sum(dump,2);
    Gsdx = bsxfun(@rdivide,dump,sumD);
else
    Gsdx = dump;
end


Gidx = repmat([1:nSmp]',1,r);
Gjdx = idx;
Z = sparse(Gidx(:),Gjdx(:),Gsdx(:),nSmp,p);



end


%  min  x'*A*x - x'*b
%  s.t. x'1=1, x>=0
function [x, obj]=SimplexQP_acc(A, b, x0)


NIter = 500;
NStop = 20;

[n] = size(A,1);
if nargin < 3
    x = 1/n*ones(n,1);
else
    x = x0;
end;

x1 = x; 
t = 1;
t1 = 0;
r = 0.5;  % r=1/mu;
%obj = zeros(NIter,1);
for iter = 1:NIter
    p = (t1-1)/t;
    s = x + p*(x-x1); 
    x1 = x;
    g = 2*A*s - b;
    ob1 = x'*A*x - x'*b;
    for it = 1:NStop
        z = s - r*g;
        z = EProjSimplex_new(z,1);  % z'1=1;z>=0; z=alpha;
        ob = z'*A*z - z'*b;
        if ob1 < ob
            r = 0.5*r; % rho=2;
        else
            break;
        end
    end
    if it == NStop
        obj(iter) = ob;
        %disp('not');
        break;
    end
    x = z;
    t1 = t;
    t = (1+sqrt(1+4*t^2))/2;
    
    
    obj(iter) = ob;
end
end






function [x, ft] = EProjSimplex_new(v, k)

%
%% Problem
%
%  min  1/2 || x - v||^2
%  s.t. x>=0, 1'x=k
%

if nargin < 2
    k = 1;
end;

ft=1;
n = length(v);

v0 = v-mean(v) + k/n;
%vmax = max(v0);
vmin = min(v0);
if vmin < 0
    f = 1;
    lambda_m = 0;
    while abs(f) > 10^-10
        v1 = v0 - lambda_m;
        posidx = v1>0;
        npos = sum(posidx);
        g = -npos;
        f = sum(v1(posidx)) - k;
        lambda_m = lambda_m - f/g;
        ft=ft+1;
        if ft > 100
            x = max(v1,0);
            break;
        end;
    end;
    x = max(v1,0);

else
    x = v0;
end    
end

% The non-paralleled version
function B = updateB(X,A,B,Y,H,lambda,N,K)
Q = H * (Y' * Y + eps * eye(K))^-0.5 * Y';
HH = 0;
ff = 0;

HH = HH +  A' * A;
%HH = HH + delta(v)^2 * eye(M);
ff = ff - 2  * A' * X;

HH = 2 * HH;
HH = (HH + HH')/2;
ff = ff - lambda * Q;

for i = 1:N
    B(i,:) = SimplexQP_acc(HH / 2, -ff(:,i));
end
end

% The paralleled version
function B = updateB_par(X,A,B,Y,H,lambda,N,K)
Q = H * (Y' * Y + eps * eye(K))^-0.5 * Y';
HH = 0;
ff = 0;

HH = HH +  A' * A;
%HH = HH + delta(v)^2 * eye(M);
ff = ff - 2  * A' * X;

HH = 2 * HH;
HH = (HH + HH')/2;
ff = ff - lambda * Q;

parfor i = 1:N
    B(i,:) = SimplexQP_acc(HH / 2, -ff(:,i));
end
end

function Y = updateY_my(Y,G, N, K)
yg = diag(Y'* G)'; 
yy = diag(Y'*Y+eps*eye(K))'; 

for i = 1:N
    gi = G(i,:);
    yi = Y(i,:);
    si = (yg+gi.*(1-yi))./sqrt(yy+1-yi) - (yg-gi.*yi)./sqrt(yy-yi);
    [~,index] = max(si(:));
    old1_idx = find(yi == 1,1);
    
    Y(i,old1_idx) = 0;
    Y(i,index) = 1;  
    
    
    yg(old1_idx) = yg(old1_idx) - G(i,old1_idx);    
    yg(index) = yg(index) + G(i,index);
    
    yy(old1_idx) = yy(old1_idx) - 1;
    yy(index) = yy(index) + 1;
    
    
end
end


function [label, center, bCon, sumD, D] = litekmeans(X, k, varargin)
%LITEKMEANS K-means clustering, accelerated by matlab matrix operations.
%
%   label = LITEKMEANS(X, K) partitions the points in the N-by-P data matrix
%   X into K clusters.  This partition minimizes the sum, over all
%   clusters, of the within-cluster sums of point-to-cluster-centroid
%   distances.  Rows of X correspond to points, columns correspond to
%   variables.  KMEANS returns an N-by-1 vector label containing the
%   cluster indices of each point.
%
%   [label, center] = LITEKMEANS(X, K) returns the K cluster centroid
%   locations in the K-by-P matrix center.
%
%   [label, center, bCon] = LITEKMEANS(X, K) returns the bool value bCon to
%   indicate whether the iteration is converged.  
%
%   [label, center, bCon, SUMD] = LITEKMEANS(X, K) returns the
%   within-cluster sums of point-to-centroid distances in the 1-by-K vector
%   sumD.    
%
%   [label, center, bCon, SUMD, D] = LITEKMEANS(X, K) returns
%   distances from each point to every centroid in the N-by-K matrix D. 
%
%   [ ... ] = LITEKMEANS(..., 'PARAM1',val1, 'PARAM2',val2, ...) specifies
%   optional parameter name/value pairs to control the iterative algorithm
%   used by KMEANS.  Parameters are:
%
%   'Distance' - Distance measure, in P-dimensional space, that KMEANS
%      should minimize with respect to.  Choices are:
%            {'sqEuclidean'} - Squared Euclidean distance (the default)
%             'cosine'       - One minus the cosine of the included angle
%                              between points (treated as vectors). Each
%                              row of X SHOULD be normalized to unit. If
%                              the intial center matrix is provided, it
%                              SHOULD also be normalized.
%
%   'Start' - Method used to choose initial cluster centroid positions,
%      sometimes known as "seeds".  Choices are:
%         {'sample'}  - Select K observations from X at random (the default)
%           matrix   - A K-by-P matrix of starting locations; or a K-by-1
%                      indicate vector indicating which K points in X
%                      should be used as the initial center.  In this case,
%                      you can pass in [] for K, and KMEANS infers K from
%                      the first dimension of the matrix.
%
%   'MaxIter'  - Maximum number of iterations allowed.  Default is 100.
%
%   'Replicates' - Number of times to repeat the clustering, each with a
%                  new set of initial centroids. Default is 1. If the
%                  initial centroids are provided, the replicate will be
%                  automatically set to be 1.
%
%
%
%    Examples:
%
%       fea = rand(500,10);
%       [label, center] = litekmeans(fea, 5, 'MaxIter', 50);
%
%       fea = rand(500,10);
%       [label, center] = litekmeans(fea, 5, 'MaxIter', 50, 'Replicates', 10);
%
%       fea = rand(500,10);
%       [label, center, bCon, sumD, D] = litekmeans(fea, 5, 'MaxIter', 50);
%       TSD = sum(sumD);
%
%       fea = rand(500,10);
%       initcenter = rand(5,10);
%       [label, center] = litekmeans(fea, 5, 'MaxIter', 50, 'Start', initcenter);
%
%       fea = rand(500,10);
%       idx=randperm(500);
%       [label, center] = litekmeans(fea, 5, 'MaxIter', 50, 'Start', idx(1:5));
%
%
%   See also KMEANS
%
%   version 2.0 --December/2011
%   version 1.0 --November/2011
%
%   Written by Deng Cai (dengcai AT gmail.com)


if nargin < 2
    error('litekmeans:TooFewInputs','At least two input arguments required.');
end

[n, p] = size(X);


pnames = {   'distance' 'start'   'maxiter'  'replicates' 'onlinephase'};
dflts =  {'sqeuclidean' 'sample'       []        []      'off'  };
[eid,errmsg,distance,start,maxit,reps,online] = getargs(pnames, dflts, varargin{:});
if ~isempty(eid)
    error(sprintf('litekmeans:%s',eid),errmsg);
end

if ischar(distance)
    distNames = {'sqeuclidean','cosine'};
    j = strcmpi(distance, distNames);
    j = find(j);
    if length(j) > 1
        error('litekmeans:AmbiguousDistance', ...
            'Ambiguous ''Distance'' parameter value:  %s.', distance);
    elseif isempty(j)
        error('litekmeans:UnknownDistance', ...
            'Unknown ''Distance'' parameter value:  %s.', distance);
    end
    distance = distNames{j};
else
    error('litekmeans:InvalidDistance', ...
        'The ''Distance'' parameter value must be a string.');
end


center = [];
if ischar(start)
    startNames = {'sample','cluster'};
    j = find(strncmpi(start,startNames,length(start)));
    if length(j) > 1
        error(message('litekmeans:AmbiguousStart', start));
    elseif isempty(j)
        error(message('litekmeans:UnknownStart', start));
    elseif isempty(k)
        error('litekmeans:MissingK', ...
            'You must specify the number of clusters, K.');
    end
    if j == 2
        if floor(.1*n) < 5*k
            j = 1;
        end
    end
    start = startNames{j};
elseif isnumeric(start)
    if size(start,2) == p
        center = start;
    elseif (size(start,2) == 1 || size(start,1) == 1)
        center = X(start,:);
    else
        error('litekmeans:MisshapedStart', ...
            'The ''Start'' matrix must have the same number of columns as X.');
    end
    if isempty(k)
        k = size(center,1);
    elseif (k ~= size(center,1))
        error('litekmeans:MisshapedStart', ...
            'The ''Start'' matrix must have K rows.');
    end
    start = 'numeric';
else
    error('litekmeans:InvalidStart', ...
        'The ''Start'' parameter value must be a string or a numeric matrix or array.');
end

% The maximum iteration number is default 100
if isempty(maxit)
    maxit = 100;
end

% Assume one replicate
if isempty(reps) || ~isempty(center)
    reps = 1;
end

if ~(isscalar(k) && isnumeric(k) && isreal(k) && k > 0 && (round(k)==k))
    error('litekmeans:InvalidK', ...
        'X must be a positive integer value.');
elseif n < k
    error('litekmeans:TooManyClusters', ...
        'X must have more rows than the number of clusters.');
end


bestlabel = [];
sumD = zeros(1,k);
bCon = false;

for t=1:reps
    switch start
        case 'sample'
            center = X(randsample(n,k),:);
        case 'cluster'
            Xsubset = X(randsample(n,floor(.1*n)),:);
            [dump, center] = litekmeans(Xsubset, k, varargin{:}, 'start','sample', 'replicates',1);
        case 'numeric'
    end
    
    last = 0;label=1;
    it=0;
    
    switch distance
        case 'sqeuclidean'
            while any(label ~= last) && it<maxit
                last = label;
                
                bb = full(sum(center.*center,2)');
                ab = full(X*center');
                D = bb(ones(1,n),:) - 2*ab;
                
                [val,label] = min(D,[],2); % assign samples to the nearest centers
                ll = unique(label);
                if length(ll) < k
                    %disp([num2str(k-length(ll)),' clusters dropped at iter ',num2str(it)]);
                    missCluster = 1:k;
                    missCluster(ll) = [];
                    missNum = length(missCluster);
                    
                    aa = sum(X.*X,2);
                    val = aa + val;
                    [dump,idx] = sort(val,1,'descend');
                    label(idx(1:missNum)) = missCluster;
                end
                E = sparse(1:n,label,1,n,k,n);  % transform label into indicator matrix
                center = full((E*spdiags(1./sum(E,1)',0,k,k))'*X);    % compute center of each cluster
                it=it+1;
            end
            if it<maxit
                bCon = true;
            end
            if isempty(bestlabel)
                bestlabel = label;
                bestcenter = center;
                if reps>1
                    if it>=maxit
                        aa = full(sum(X.*X,2));
                        bb = full(sum(center.*center,2));
                        ab = full(X*center');
                        D = bsxfun(@plus,aa,bb') - 2*ab;
                        D(D<0) = 0;
                    else
                        aa = full(sum(X.*X,2));
                        D = aa(:,ones(1,k)) + D;
                        D(D<0) = 0;
                    end
                    D = sqrt(D);
                    for j = 1:k
                        sumD(j) = sum(D(label==j,j));
                    end
                    bestsumD = sumD;
                    bestD = D;
                end
            else
                if it>=maxit
                    aa = full(sum(X.*X,2));
                    bb = full(sum(center.*center,2));
                    ab = full(X*center');
                    D = bsxfun(@plus,aa,bb') - 2*ab;
                    D(D<0) = 0;
                else
                    aa = full(sum(X.*X,2));
                    D = aa(:,ones(1,k)) + D;
                    D(D<0) = 0;
                end
                D = sqrt(D);
                for j = 1:k
                    sumD(j) = sum(D(label==j,j));
                end
                if sum(sumD) < sum(bestsumD)
                    bestlabel = label;
                    bestcenter = center;
                    bestsumD = sumD;
                    bestD = D;
                end
            end
        case 'cosine'
            while any(label ~= last) && it<maxit
                last = label;
                W=full(X*center');
                [val,label] = max(W,[],2); % assign samples to the nearest centers
                ll = unique(label);
                if length(ll) < k
                    missCluster = 1:k;
                    missCluster(ll) = [];
                    missNum = length(missCluster);
                    [dump,idx] = sort(val);
                    label(idx(1:missNum)) = missCluster;
                end
                E = sparse(1:n,label,1,n,k,n);  % transform label into indicator matrix
                center = full((E*spdiags(1./sum(E,1)',0,k,k))'*X);    % compute center of each cluster
                centernorm = sqrt(sum(center.^2, 2));
                center = center ./ centernorm(:,ones(1,p));
                it=it+1;
            end
            if it<maxit
                bCon = true;
            end
            if isempty(bestlabel)
                bestlabel = label;
                bestcenter = center;
                if reps>1
                    if any(label ~= last)
                        W=full(X*center');
                    end
                    D = 1-W;
                    for j = 1:k
                        sumD(j) = sum(D(label==j,j));
                    end
                    bestsumD = sumD;
                    bestD = D;
                end
            else
                if any(label ~= last)
                    W=full(X*center');
                end
                D = 1-W;
                for j = 1:k
                    sumD(j) = sum(D(label==j,j));
                end
                if sum(sumD) < sum(bestsumD)
                    bestlabel = label;
                    bestcenter = center;
                    bestsumD = sumD;
                    bestD = D;
                end
            end
    end
end

label = bestlabel;
center = bestcenter;
if reps>1
    sumD = bestsumD;
    D = bestD;
elseif nargout > 3
    switch distance
        case 'sqeuclidean'
            if it>=maxit
                aa = full(sum(X.*X,2));
                bb = full(sum(center.*center,2));
                ab = full(X*center');
                D = bsxfun(@plus,aa,bb') - 2*ab;
                D(D<0) = 0;
            else
                aa = full(sum(X.*X,2));
                D = aa(:,ones(1,k)) + D;
                D(D<0) = 0;
            end
            D = sqrt(D);
        case 'cosine'
            if it>=maxit
                W=full(X*center');
            end
            D = 1-W;
    end
    for j = 1:k
        sumD(j) = sum(D(label==j,j));
    end
end

end
function [eid,emsg,varargout]=getargs(pnames,dflts,varargin)
%GETARGS Process parameter name/value pairs 
%   [EID,EMSG,A,B,...]=GETARGS(PNAMES,DFLTS,'NAME1',VAL1,'NAME2',VAL2,...)
%   accepts a cell array PNAMES of valid parameter names, a cell array
%   DFLTS of default values for the parameters named in PNAMES, and
%   additional parameter name/value pairs.  Returns parameter values A,B,...
%   in the same order as the names in PNAMES.  Outputs corresponding to
%   entries in PNAMES that are not specified in the name/value pairs are
%   set to the corresponding value from DFLTS.  If nargout is equal to
%   length(PNAMES)+1, then unrecognized name/value pairs are an error.  If
%   nargout is equal to length(PNAMES)+2, then all unrecognized name/value
%   pairs are returned in a single cell array following any other outputs.
%
%   EID and EMSG are empty if the arguments are valid.  If an error occurs,
%   EMSG is the text of an error message and EID is the final component
%   of an error message id.  GETARGS does not actually throw any errors,
%   but rather returns EID and EMSG so that the caller may throw the error.
%   Outputs will be partially processed after an error occurs.
%
%   This utility can be used for processing name/value pair arguments.
%
%   Example:
%       pnames = {'color' 'linestyle', 'linewidth'}
%       dflts  = {    'r'         '_'          '1'}
%       varargin = {{'linew' 2 'nonesuch' [1 2 3] 'linestyle' ':'}
%       [eid,emsg,c,ls,lw] = statgetargs(pnames,dflts,varargin{:})    % error
%       [eid,emsg,c,ls,lw,ur] = statgetargs(pnames,dflts,varargin{:}) % ok

% We always create (nparams+2) outputs:
%    one each for emsg and eid
%    nparams varargs for values corresponding to names in pnames
% If they ask for one more (nargout == nparams+3), it's for unrecognized
% names/values

%   Original Copyright 1993-2008 The MathWorks, Inc. 
%   Modified by Deng Cai (dengcai@gmail.com) 2011.11.27

% Initialize some variables
emsg = '';
eid = '';
nparams = length(pnames);
varargout = dflts;
unrecog = {};
nargs = length(varargin);

% Must have name/value pairs
if mod(nargs,2)~=0
    eid = 'WrongNumberArgs';
    emsg = 'Wrong number of arguments.';
else
    % Process name/value pairs
    for j=1:2:nargs
        pname = varargin{j};
        if ~ischar(pname)
            eid = 'BadParamName';
            emsg = 'Parameter name must be text.';
            break;
        end
        i = strcmpi(pname,pnames);
        i = find(i);
        if isempty(i)
            % if they've asked to get back unrecognized names/values, add this
            % one to the list
            if nargout > nparams+2
                unrecog((end+1):(end+2)) = {varargin{j} varargin{j+1}};
                % otherwise, it's an error
            else
                eid = 'BadParamName';
                emsg = sprintf('Invalid parameter name:  %s.',pname);
                break;
            end
        elseif length(i)>1
            eid = 'BadParamName';
            emsg = sprintf('Ambiguous parameter name:  %s.',pname);
            break;
        else
            varargout{i} = varargin{j+1};
        end
    end
end

varargout{nparams+1} = unrecog;
end
