function A = readlas(infilename) % 改編自LASreadALL
%% Open the file
fid = fopen(infilename);

% Check whether the file is valid
if fid == -1
    error('Error opening file')
end


%% Read in important information from the header
% Check whether the LAS format is 1.1
fseek(fid, 24, 'bof');
VersionMajor = fread(fid,1,'uchar');
VersionMinor = fread(fid,1,'uchar');

% afarris2011Aug20 changed the following line to read LAS1.2 files
% if VersionMajor ~= 1 || VersionMinor ~= 1 
if VersionMajor ~= 1  
    error('LAS format is not 1.*')
end

% Read in the offset to point data
fseek(fid, 96, 'bof');
OffsetToPointData = fread(fid,1,'uint32');
 
% Read in the point data fotmat ID
fseek(fid, 104, 'bof');
pointDataFormatID = fread(fid,1,'uchar');
 
% Read in the point data record length
fseek(fid, 105, 'bof');
pointDataRecordLength = fread(fid,1,'short');

% The number of bytes from the beginning of the file to the first point record
% data field is used to access the attributes of the point data
c = OffsetToPointData;
 
% The number of bytes to skip after reading in each value is based on
% 'pointDataRecordLength' So I need a short version of the variable name:
p = pointDataRecordLength;

% Read in the scale factors and offsets required to calculate the coordinates
fseek(fid, 131, 'bof');
XScaleFactor = fread(fid,1,'double');
YScaleFactor = fread(fid,1,'double');
ZScaleFactor = fread(fid,1,'double');
XOffset = fread(fid,1,'double');
YOffset = fread(fid,1,'double');
ZOffset = fread(fid,1,'double');
 
% The number of bytes from the beginning of the file to the first point record
% data field is used to access the attributes of the point data
c = OffsetToPointData;
 
% The number of bytes to skip after reading in each value is based on
% 'pointDataRecordLength' So I need a short version of the variable name:
p = pointDataRecordLength;
 
%% Now read in the data
 
% Reads in the X coordinates of the points;  making use of the
% XScaleFactor and XOffset values in the header.
fseek(fid, c, 'bof');
X1 = fread(fid,inf,'int32',p-4);
A.X = X1*XScaleFactor+XOffset;
 
% Read in the Y coordinates of the points
fseek(fid, c+4, 'bof');
Y1 = fread(fid,inf,'int32',p-4);
A.Y = Y1*YScaleFactor+YOffset;
 
% Read in the Z coordinates of the points
fseek(fid, c+8, 'bof');
Z1 = fread(fid,inf,'int32',p-4);
A.Z = Z1*ZScaleFactor+ZOffset;

% Read in the Intensity values of the points
fseek(fid, c+12, 'bof');
A.intensity = fread(fid,inf,'uint16',p-2);

% Read in color
fseek(fid, c+28, 'bof');
A.R = fread(fid,inf,'short',p-2);
fseek(fid, c+30, 'bof');
A.G = fread(fid,inf,'short',p-2);
fseek(fid, c+32, 'bof');
A.B = fread(fid,inf,'short',p-2);
% modify RGB value
A.R(A.R<0) = A.R(A.R<0) + 65536;
A.G(A.G<0) = A.G(A.G<0) + 65536;
A.B(A.B<0) = A.R(A.B<0) + 65536;

% Read in classification
fseek(fid, c+15, 'bof');
A.classification = fread(fid,inf,'char',p-1);

if pointDataRecordLength ~= 34
    error('pointDataRecordLength is not expected (expected: 34)')
end

% las = [A.X, A.Y, A.Z, A.intensity, A.R, A.G, A.B, A.classification ];
end