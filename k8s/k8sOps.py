from kubernetes import client,config,utils
from kubernetes.client.rest import ApiException
import yaml

config.load_kube_config()
k8s_client = client.ApiClient()
apps_v1 = client.AppsV1Api()
core_v1 = client.CoreV1Api()

def read_yaml_all(yaml_path):
    try:
        with open(yaml_path,"r",encoding="utf-8") as f:
            data = yaml.load(f,Loader=yaml.FullLoader)
            return data
    except:
        return None

def apply_from_single_file(yaml_file : str):
    data = read_yaml_all(yaml_file)
    name = data["metadata"]["name"]
    namespace = data["metadata"]["namespace"]
    if deployment_exists(name, namespace):
        print("Deployment already exists")
    else:
        utils.create_from_yaml(k8s_client,yaml_file,verbose=True)
    return

def deployment_exists(name : str, namespace = "default"):
    resp = apps_v1.list_namespaced_deployment(namespace=namespace)
    for i in resp.items:
        if i.metadata.name == name:
            return True
    return False

def service_exists(name : str, namespace = "default"):
    resp = core_v1.list_namespaced_service(namespace=namespace)
    for i in resp.items:
        if i.metadata.name == name:
            return True
    return False

def namespace_exists(name : str):
    try:
        core_v1.read_namespace(name)
        return True
    except ApiException as e:
        return False

def create_namespace(name = "defaultz"):
    metadata = client.V1ObjectMeta(name="serverless")
    body = client.V1Namespace(metadata=metadata) # V1Namespace |
    try:
        api_response = core_v1.create_namespace(body, pretty=True)
        print("Namespace created. Status='%s'" % str(api_response.status))
    except ApiException as e:
        print("Exception when creating namespace: %s\n" % e)
    return

if __name__ == "__main__":
    pass
