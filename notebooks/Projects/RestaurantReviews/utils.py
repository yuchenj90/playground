import json, requests, os, time, aiohttp, asyncio
from dotenv import load_dotenv
load_dotenv()
API_KEY = os.getenv("GOOGLE_PLACE_API_KEY")

def get_nearby_restaurants(lat, lng, rad="10000", stype="restaurant"):
    params = f"location={lat}%2C{lng}&radius={rad}&type={stype}&rankby=prominence&key={API_KEY}"
    URL_NearbySearch = f"https://maps.googleapis.com/maps/api/place/nearbysearch/json?{params}"
    r = requests.get(URL_NearbySearch)
    data = json.loads(r.content)
    return data['results']

def get_reviews(d, place_id):
    res = []
    for reviews_sort in ['most_relevant', 'newest']:
        params = f"place_id={place_id}&reviews_sort={reviews_sort}&key={API_KEY}"
        r = requests.get(f"https://maps.googleapis.com/maps/api/place/details/json?{params}")
        res.extend(json.loads(r.content)['result']['reviews'])
    return d, res

'''
def get_restaurants_parallel(lat, lng, limit = 1000, max_worker = 50):
    def post_process(result):
        for d in result:
            if d['place_id'] not in visited:
                queue.append({'name': d['name'], 'place_id': d['place_id'], 
                              'lat': d['geometry']['location']['lat'], 'lng': d['geometry']['location']['lng']})
                visited[d['place_id']] = True
                
    queue = [{'name': '', 'place_id': '', 'lat': lat, 'lng': lng}]
    visited = {}
    p = 0
    while p < len(queue):
        x = queue[p]
        p_next = min([len(queue), p+max_worker])
        pool = Pool(max_worker)
        for i in range(p, p_next):
            pool_res = pool.apply_async(get_nearby_restaurants, args=(queue[i]['lat'], queue[i]['lng']), callback=post_process)
        pool.close()
        pool.join()
        p = p_next
        if len(queue) > limit:
            break
    return queue

def get_all_reviews(data, max_worker=50):
    def add_res(x):
        x[0]['reviews'] = x[1]
        
    pool = Pool(max_worker)
    res = []
    for d in data:
        pool.apply_async(get_reviews, args=(d, d['place_id']), callback=add_res)
    pool.close()
    pool.join()
    return data
'''
# Asynchronous function to fetch data from a given URL using aiohttp
async def fetch_data(url):
    async with aiohttp.ClientSession() as session:
        # Use 'session.get()' to make an asynchronous HTTP GET request
        async with session.get(url) as response:
            return await response.json()

async def get_batch_restaurants(data, rad="10000"):
    urls = []
    for d in data:
        params = f"location={d['lat']}%2C{d['lng']}&radius={rad}&type=restaurant&rankby=prominence&key={API_KEY}"
        urls.append(f"https://maps.googleapis.com/maps/api/place/nearbysearch/json?{params}")
    tasks = [fetch_data(url) for url in urls]
    res = []
    for x in await asyncio.gather(*tasks):
        res.extend(x['results'])
    return res

async def get_restaurants_parallel(lat, lng, limit = 1000, max_batchsize = 50):
    queue = [{'name': '', 'place_id': '', 'lat': lat, 'lng': lng}]
    visited = {}
    p = 0
    while p < len(queue):
        p_next = min([len(queue), p + max_batchsize])
        res = await get_batch_restaurants(queue[p:p_next])
        for x in res:
            if x['place_id'] not in visited:
                queue.append({'name': x['name'], 'place_id': x['place_id'], 
                              'lat': x['geometry']['location']['lat'], 'lng': x['geometry']['location']['lng']})
                visited[x['place_id']] = True
        if len(queue) > limit:
            break
        p = p_next
    return queue

async def fetch(session, url, semaphore):
    async with semaphore:
        async with session.get(url) as response:
            return await response.json()
        
async def get_all_reviews(data, max_requests_per_second = 5):
    urls = []
    for d in data:
        for reviews_sort in ['most_relevant', 'newest']:
            params = f"place_id={d['place_id']}&reviews_sort={reviews_sort}&key={API_KEY}"
            urls.append(f"https://maps.googleapis.com/maps/api/place/details/json?{params}")

    res = []
    semaphore = asyncio.Semaphore(max_requests_per_second)
    async with aiohttp.ClientSession() as session:
        # Create a list of tasks, where each task is a call to 'fetch_data' with a specific URL
        tasks = [fetch(session, url, semaphore) for url in urls]
        # Use 'asyncio.gather()' to run the tasks concurrently and gather their results
        results = await asyncio.gather(*tasks)
        res.extend(results)
    return res
