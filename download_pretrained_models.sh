mkdir -p ckpts

echo "Download yelp_w8.tar.gz"

wget "https://cuhko365-my.sharepoint.com/:u:/g/personal/218019026_link_cuhk_edu_cn/EQn-M2lPFCJFvAsJPNA6YxQBRt8ejB4qoZnOgbpF2un2jw?e=wVEKFb&download=1" -O ckpts/yelp_w8.tar.gz

cd ckpts
# tar -zxvf base_yelp.tar.gz
# tar -zxvf large_yelp.tar.gz
# tar -zxvf large_amazon.tar.gz

tar -zxvf yelp_w8.tar.gz
cd ..
