#!/bin/bash

#   Author: Angela Crabtree

# This script automates the download of 20 PCam (patch camelyon 17) WSIs from an FTP server
# Each WSI zip file takes ~1hr to download. Zipped file size is 4-15 Gb. 
# Here's a great resource for how to download TCGA data: http://www.andrewjanowczyk.com/download-tcga-digital-pathology-images-ffpe/


# make folders to generally match nightingale directories & contents
mkdir -p ./data/tcga-dlbc
cd ./data/tcga-dlbc

# make spreadsheets 
echo "case_submitter_id	filename	stage_r	stage
TCGA-FA-A4BB	TCGA-FA-A4BB-01Z-00-DX1.2BF28C5E-B2BE-4C0A-A369-BDE825EF32AD.svs	IV	4
TCGA-FA-A7Q1	TCGA-FA-A7Q1-01Z-00-DX1.99220CAA-6160-48FA-AA65-94705EA46773.svs	I	1
TCGA-FA-A86F	TCGA-FA-A86F-01Z-00-DX1.EE36DC3F-7539-41E8-ADF6-6BA0FB0F47C4.svs	III	3
TCGA-FF-8042	TCGA-FF-8042-01Z-00-DX1.bf80d981-9209-4933-8f80-e9e689971999.svs	II	2
TCGA-FF-8047	TCGA-FF-8047-01Z-00-DX1.75aa745c-bbe3-4869-a37b-c18ee50c14d5.svs	II	2
TCGA-G8-6324	TCGA-G8-6324-01Z-00-DX1.30727896-3e02-44c8-a7b2-745f70c441ed.svs	IV	4
TCGA-G8-6325	TCGA-G8-6325-01Z-00-DX1.973b0dd6-d701-4224-9395-eab9603488ce.svs	IV	4
TCGA-G8-6907	TCGA-G8-6907-01Z-00-DX1.71ada905-9b5b-4757-8dfd-584a79d56ccc.svs	IV	4
TCGA-G8-6909	TCGA-G8-6909-01Z-00-DX1.f38a05fd-179e-40ae-9fa2-337e09235691.svs	III	3
TCGA-G8-6914	TCGA-G8-6914-01Z-00-DX1.910a33a3-3c27-4368-a223-d15a5e024c5a.svs	III	3
TCGA-GR-7353	TCGA-GR-7353-01Z-00-DX1.98554629-7e97-41ea-9fef-13e7b88f4ecc.svs	II	2
TCGA-GR-A4D5	TCGA-GR-A4D5-01Z-00-DX1.D80C72B3-C80D-47E4-AF4B-44AB5F1AD8C5.svs	IV	4
TCGA-GR-A4D6	TCGA-GR-A4D6-01Z-00-DX1.5C79ED21-5689-45C5-8E08-2EF67056F7CA.svs	I	1
TCGA-GR-A4D9	TCGA-GR-A4D9-01Z-00-DX1.443D848D-2F3A-461D-8990-2C823D783B0A.svs	IV	4
TCGA-GS-A9TQ	TCGA-GS-A9TQ-01Z-00-DX1.AC86A0E2-8FBA-4A3D-A1C8-BB403D9C93E8.svs	II	2
TCGA-GS-A9TU	TCGA-GS-A9TU-01Z-00-DX1.F10984B9-F06A-4729-B668-A5C184EE6D28.svs	II	2
TCGA-GS-A9TV	TCGA-GS-A9TV-01Z-00-DX1.FD6E85B5-1F95-47DA-84CE-C7B888D29D26.svs	II	2
TCGA-GS-A9TW	TCGA-GS-A9TW-01Z-00-DX1.D38BD003-8DF1-4740-B83D-3426B00EF686.svs	II	2
TCGA-GS-A9TX	TCGA-GS-A9TX-01Z-00-DX1.5A1B7A1F-76EE-4E63-92C1-44AD0523F37B.svs	I	1
TCGA-GS-A9TY	TCGA-GS-A9TY-01Z-00-DX1.A04FA83C-736C-4C55-91F2-B1CD892178A6.svs	I	1
TCGA-GS-A9TZ	TCGA-GS-A9TZ-01Z-00-DX1.75FFC0BE-5C16-4AFC-BC7D-42D047636DBC.svs	II	2
TCGA-GS-A9U3	TCGA-GS-A9U3-01Z-00-DX1.2A338F39-0486-45EF-B829-19360F2E2FC9.svs	III	3
TCGA-GS-A9U4	TCGA-GS-A9U4-01Z-00-DX1.E8F03678-760A-4956-A829-708C6F6B968B.svs	II	2" > meta_TCGA-DLBC.txt

# make manifest (originally downloaded from TGCA:
echo "id	filename	md5	size	state
fab9039a-7776-4420-af7f-cb861cf45bf4	TCGA-GS-A9TU-01Z-00-DX1.F10984B9-F06A-4729-B668-A5C184EE6D28.svs	3842f7922d7e0eadbe9d53b258fbd8ff	632981733	released
10017aed-76fd-446f-8ed4-466e850496c9	TCGA-G8-6907-01Z-00-DX1.71ada905-9b5b-4757-8dfd-584a79d56ccc.svs	6cc1ff4400d9f76c2308627f93a26e3a	642203197	released
62a3ed21-180b-4f25-b29d-8c88cda7a65d	TCGA-G8-6914-01Z-00-DX1.910a33a3-3c27-4368-a223-d15a5e024c5a.svs	a427655b6701007977fa0096f503cbb1	920306275	released
426f34b8-4d6e-455a-a399-3729940c1e32	TCGA-GS-A9TY-01Z-00-DX1.A04FA83C-736C-4C55-91F2-B1CD892178A6.svs	b835b2d1607535fe6e55aa7408279ce1	1528973641	released
fed295ee-f9ec-44fd-8801-4c92b61419c5	TCGA-GS-A9TQ-01Z-00-DX1.AC86A0E2-8FBA-4A3D-A1C8-BB403D9C93E8.svs	654d78191633f2393da5a55bbaca0206	1061993409	released
256336ac-ca5d-4dc4-9486-709d52bec0ec	TCGA-GS-A9TX-01Z-00-DX1.5A1B7A1F-76EE-4E63-92C1-44AD0523F37B.svs	abac1b7a7a1d3ff1d6f9052b1e055ddf	1547190635	released
cb6a8ba6-01e3-41f6-860d-2a56b1e2e9e0	TCGA-G8-6324-01Z-00-DX1.30727896-3e02-44c8-a7b2-745f70c441ed.svs	99f12d2f92c290716b771febbe43e085	711700673	released
bead2d5c-ec1d-49c5-94b7-bec0afd47098	TCGA-GR-A4D6-01Z-00-DX1.5C79ED21-5689-45C5-8E08-2EF67056F7CA.svs	e3dc98b6d3e0baf94314919ac8f38fc2	577999254	released
0609ae18-458a-4971-938e-1942e35c11b2	TCGA-FF-8047-01Z-00-DX1.75aa745c-bbe3-4869-a37b-c18ee50c14d5.svs	7cd7d28263fb257ce7fb8f3c1ea7ed83	291477431	released
bc7c89ee-de0f-4fc6-b89f-60247bdbf543	TCGA-GR-7353-01Z-00-DX1.98554629-7e97-41ea-9fef-13e7b88f4ecc.svs	7c7b8c1c2bd8501544d9ef1a0cfc06e8	157891802	released
6d532813-3280-43e8-9e1c-49080ed8f734	TCGA-G8-6909-01Z-00-DX1.f38a05fd-179e-40ae-9fa2-337e09235691.svs	964a775dedddb9f147f6aca337f5a0a4	271454999	released
4dc12562-6d21-49c0-89a0-644c952fe96f	TCGA-GR-A4D9-01Z-00-DX1.443D848D-2F3A-461D-8990-2C823D783B0A.svs	b7dea8a07d7a8212864053470587f4d2	325792520	released
2f2ae127-0a3d-46b6-8485-ec54d30ff8c3	TCGA-FA-A86F-01Z-00-DX1.EE36DC3F-7539-41E8-ADF6-6BA0FB0F47C4.svs	c54c7fa9d6e6b6f1174c386fc13b1fef	753663393	released
5dddda2b-4a82-479e-926d-12abdfd1338a	TCGA-GR-A4D5-01Z-00-DX1.D80C72B3-C80D-47E4-AF4B-44AB5F1AD8C5.svs	6bc4eb5889204899eb302c3dd6912966	491714792	released
4be97a10-3438-439f-92a1-816b35655a54	TCGA-GS-A9U3-01Z-00-DX1.2A338F39-0486-45EF-B829-19360F2E2FC9.svs	a5b8b5df070f9ecda94a9ba85f6248f7	980484625	released
cd08437a-de67-45ad-9d3d-7c547c0f03e2	TCGA-FA-A7Q1-01Z-00-DX1.99220CAA-6160-48FA-AA65-94705EA46773.svs	33728218721b1e51e20d58c696be8983	174631217	released
1e5d1461-c743-4973-a741-0aa4b93e0e7d	TCGA-GS-A9U4-01Z-00-DX1.E8F03678-760A-4956-A829-708C6F6B968B.svs	0e7fa13de6b79d14ac66eb0ad90270c9	897185589	released
1a66846d-90a9-42bd-af12-6582b92908c4	TCGA-FA-A4BB-01Z-00-DX1.2BF28C5E-B2BE-4C0A-A369-BDE825EF32AD.svs	6d8914ca33c05566d7cb5a8c0feeafb8	1089784861	released
efc52991-f3d9-495c-9597-d883b478b09e	TCGA-GS-A9TW-01Z-00-DX1.D38BD003-8DF1-4740-B83D-3426B00EF686.svs	2725504824127f32123c791c904a2568	1370021319	released
67692367-8c80-4705-a0c2-418f88018268	TCGA-G8-6325-01Z-00-DX1.973b0dd6-d701-4224-9395-eab9603488ce.svs	5b9b71a79e7ef2559a7a4386f87ef1c4	902950541	released
757c24dc-2d22-47a2-804f-f27d07696f4c	TCGA-GS-A9TZ-01Z-00-DX1.75FFC0BE-5C16-4AFC-BC7D-42D047636DBC.svs	2ea83e6a157e098af71d601a4d7ec1e3	1389587015	released
966c9041-7d78-4fd4-a0a2-bbea23623ee6	TCGA-FF-8042-01Z-00-DX1.bf80d981-9209-4933-8f80-e9e689971999.svs	2f66ee6b2384de3672ded8a74a990df2	531954549	released
191cd824-67d7-463f-b00b-abe166a4f9ea	TCGA-GS-A9TV-01Z-00-DX1.FD6E85B5-1F95-47DA-84CE-C7B888D29D26.svs	79ab5de9c8084ce3a165c49fe5054326	475116067	released" > gdc-manifest_TCGA-DLBC.txt

# download TGCA downloading app
wget https://gdc.cancer.gov/files/public/file/gdc-client_v1.6.1_Ubuntu_x64.zip
unzip gdc-client_v1.6.1_Ubuntu_x64.zip
rm gdc-client_v1.6.1_Ubuntu_x64.zip
chmod a+x gdc-client

# download WSIs (will take approximately 10-30 seconds per image)
./gdc-client download -m ./gdc-manifest_TCGA-DLBC.txt

# extract .svs files from each folder
for file in */*.svs; do mv ${file} .; done
cd ../..