
def read_response(response: str):
    response_list = response.split(' ')
    service_time = float(response_list[1][:-1])
    inference_time = float(response_list[3][:-1])
    prepocess_time = float(response_list[5][:-1])
    return service_time, inference_time, prepocess_time