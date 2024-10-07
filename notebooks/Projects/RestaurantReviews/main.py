from utils import get_restaurants_parallel, get_all_reviews
import json, time, aiohttp, asyncio

async def main():
    lat = "34.0200393"
    lng = "-118.7413648"
    data_filename = "restaurants_info.json"
    N_res = 1000
    
    print("-------------------Getting nearby restaurants-------------------")
    start_time = time.time()
    data = await get_restaurants_parallel(lat, lng, limit=N_res, max_batchsize=50)
    print(f"Finished in {(time.time() - start_time)} seconds. Got {len(data)-1} restaurants")

    print("-------------------Getting restaurants reviews-------------------")
    start_time = time.time()
    res = await get_all_reviews(data[1:], max_requests_per_second=5)
    print(f"Finished in {(time.time() - start_time)} seconds.")
    
    print("-------------------Data post-processing-------------------")
    
    place_dict = {}
    for d in data[1:]:
        place_dict[d['place_id']] = d.copy()
    for x in res:
        if 'reviews' in x['result']:
            if 'reviews' not in place_dict[x['result']['place_id']]:
                place_dict[x['result']['place_id']]['reviews'] = x['result']['reviews']
            else:
                place_dict[x['result']['place_id']]['reviews'].extend(x['result']['reviews'])
    restaurants_info = list(place_dict.values())
    with open(data_filename, 'w') as f:
        json.dump(restaurants_info, f, indent=2)
    print(f"Finished data post-processing. Data saved to {data_filename}")
