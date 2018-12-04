FILE_ID="1c8pCh723Z4UX8OVbOPKTWTAgDw_Qh-e9"
FILE_NAME="tmp/sample.txt"
curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=${FILE_ID}" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=${FILE_ID}" -o ${FILE_NAME}
