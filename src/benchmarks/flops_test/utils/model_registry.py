import gto
import dvc.api
from tabulate import tabulate
from collections import defaultdict
from tqdm import tqdm
import subprocess
import os.path as osp
import os
import argparse
import shutil
import pandas as pd
import yaml

class ModelRegistry:
    def __init__(self, repo_path: str='.'):
        self.repo_path = repo_path # Specify the repository path; '.' refers to the current directory
        self.extras = {'info': False, 'onnx': False, 'onnx_model-card': False}

    def check_extras(self, extras: dict, artifact_name: str, file_path: str="dvc.yaml") -> dict:
        """
        Check if the extras are available in the model registry .yaml file.
        """
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)
        for extra in extras:
            artifact = artifact_name+ ("_"+extra if extra != '' else '')
            if  artifact not in data['artifacts']:
                extras[extra] = False
                print(f"Warning: Artifact '{artifact}' is not available in '{file_path}'")
        return extras
        

    def fetch_models(self) -> dict:
        """
        Fetch the models from the model registry.
        """
        artifacts_info = gto.api.show(repo=self.repo_path, name='')
        artifacts_list = [name for name, _ in artifacts_info.items()]

        models_data = defaultdict(dict)
        for artifact_name in tqdm(artifacts_list, desc='Fetching models data'):
            artifacts_info = gto.api.show(repo=self.repo_path, name=artifact_name)
            for artifact_data in artifacts_info:
                try:
                    models_data[artifact_name]['version'].append(str(artifact_data['version'])+'\n')
                except:
                    models_data[artifact_name]['version']=[(artifact_data['version'])+'\n']

        return models_data
    
    def merge_models_data(self, models_data: dict) -> pd.DataFrame:
        """
        Merge the models data into a DataFrame.
        """
        rows = []
        for name, details in models_data.items():
            base_name = '_'.join(name.split("_")[:-1])
            extension = name.split("_")[-1]
            if extension in self.extras.keys():
                for version in details["version"]:
                    version = version.strip()
                    rows.append({
                        'Artifact': base_name,
                        'Version': version,
                        'Extra': extension.capitalize(),
                        'Exists': True
                    })
            else:
                for version in details["version"]:
                    version = version.strip()
                    rows.append({
                        'Artifact': name,
                        'Version': version,
                        'Extra': '',
                        'Exists': True
                    })
        df = pd.DataFrame(rows)
        pivot_df = df.pivot_table(index=['Artifact', 'Version'], columns='Extra', values='Exists', fill_value=False)
        pivot_df = pivot_df.reset_index().rename_axis(None, axis=1)
        pivot_df = pivot_df.replace({True: 'Yes', False: 'No'}).replace({0: 'No', 1: 'Yes'})

        merged_df = pivot_df.groupby(['Artifact']).agg(lambda x: '\n'.join(x)).drop(columns=[''])
        merged_df.index.name = 'Artifacts'

        return merged_df

    def show_models(self):
        """
        Show the models in the models int the terminal.
        """
        models_data = self.fetch_models()
        models_data = self.merge_models_data(models_data)

        headers = [models_data.index.name]+models_data.columns.tolist()
        
        print(tabulate(models_data, headers=headers, tablefmt='grid'))

    def download_model(self, artifact: str, version: str, remote: str=None, remote_config: dict=None, submodule: bool=False):
        """
        Download the model from the model registry.
        """
        artifact = artifact.replace('=', ':')
        if submodule and ':' in artifact:
            artifact_foder, artifact_name = artifact.split(':')
            submodule_repo = osp.join(self.repo_path, artifact_foder)
            cmd = f'dvc artifacts get {submodule_repo} {artifact_name} --rev {version}'
        else:
            artifact_name = artifact.split(':')[-1]
            cmd = f'dvc artifacts get {self.repo_path} {artifact} --rev {version}'

        if remote:
            cmd += f" --remote {remote}"
            if remote_config:
                remote_config_str = ' '.join([f'{key}={value}' for key, value in remote_config.items()])
                cmd += f" --remote-config {remote_config_str}"
        #print(cmd)
        try:
            return_val = subprocess.check_output(cmd, shell=True)
            downloaded_file = return_val.decode('utf-8').strip().split(' ')[-1].replace('\'','')
            _, downloaded_file_extension = osp.splitext(downloaded_file)
            downloaded_artifact_name = artifact_name+'@'+version+downloaded_file_extension
            shutil.move(osp.join(os.getcwd(),downloaded_file), osp.join(os.getcwd(),downloaded_artifact_name))
            print(f"Model downloaded to: {osp.join(os.getcwd(),downloaded_artifact_name)}")
        except subprocess.CalledProcessError as e:
            print(f"Warning: Error while trying to download the model {artifact} with version {version}")
            print(f"For legazy models (non-subrepros) use 'pip3 install dvc[ssh]==3.31.0' and 'pip3 install pygit2==1.14.1'")

    def add_model(self, artifact: str, version: str, artifact_folder:str='', push:bool=True, submodule:bool=False):
        """
        Add the model to the model registry.
        """
        if submodule: # Adds the model to the submodule and the main repo
            if not ':' in artifact:
                artifact_name = artifact
                cwd = os.getcwd()
                artifact_folder = "/".join(cwd.strip("/").split("/")[-2:])
                artifact = artifact_folder+':'+artifact_name
                parent_repo = osp.join(self.repo_path, '..', '..')
                submodule_repo = self.repo_path
            else:
                submodule_repo, artifact_name = artifact.split(':')
                parent_repo = self.repo_path

            cmd = f'gto register {artifact_name} --repo {submodule_repo} --ver {version}' + (' --push' if push else '') + ' && ' + \
                  f'gto register {artifact} --repo {parent_repo} --ver {version}' + (' --push' if push else '')

        else: # Adds the model to the main repo
            cmd = f'gto register {artifact} --repo {self.repo_path} --ver {version}' + (' --push' if push else '')
        
        try: 
            subprocess.run(cmd, shell=True)
            print(f'Model added succesfully to the registry: {artifact.replace(":","=")}@{version}')
        except:
            print('Git ssh credential not found, run the following command to push the tag manually:')
            print(f'git push origin {artifact.replace(":","=")}@{version}')            


if __name__ == '__main__':
    argsparse = argparse.ArgumentParser()
    argsparse.add_argument('action', type=str, choices=['show', 'get', 'add'], help='Models need to be added with the relative path to to the dvc.yaml file, ex: rel_path/dvc.yaml:model')
    argsparse.add_argument('--submodule', action='store_true', default=False)
    argsparse.add_argument('--repo', type=str, default='.')
    argsparse.add_argument('--model', type=str)
    argsparse.add_argument('--version', type=str)
    argsparse.add_argument('--remote', type=str, default=None, required=False)
    argsparse.add_argument('--remote-config', type=dict, default=None)
    argsparse.add_argument('--push', action='store_true', default=False)
    argsparse.add_argument('--onnx-card', action='store_true', default=False)
    argsparse.add_argument('--info', action='store_true', default=False)
    argsparse.add_argument('--onnx', action='store_true', default=False)
    argsparse.add_argument('--all-extras', action='store_true', default=False)
    argsparse.add_argument('--debug', action='store_true', default=False)

    args = argsparse.parse_args()
    
    if args.action in ['add', 'get'] and not args.model:
        argsparse.error(f"the '--model' argument is required for '{args.action}' actions")
    if args.action in ['add', 'get'] and not args.version:
        argsparse.error(f"the '--version' argument is required for '{args.action}' actions")

    extras = {'': True}
    if args.info or args.all_extras:
        extras['info'] = True
    if args.onnx or args.all_extras:
        extras['onnx'] = True
    if args.onnx_card or args.all_extras:
        extras['onnx_model-card'] = True

    model_registry = ModelRegistry(repo_path=args.repo)

    if args.action == 'show':   
        model_registry.show_models()

    if args.action == 'add':
        dvc_yaml_path = ""
        if os.path.exists(args.repo):
            dvc_yaml_path = osp.join(dvc_yaml_path, args.repo)
        if ":" in args.model:
            dvc_yaml_path = osp.join(dvc_yaml_path,args.model.split(':')[0])
        
        dvc_yaml_path = osp.join(dvc_yaml_path, 'dvc.yaml')
        
        extras = model_registry.check_extras(extras, args.model.split(':')[-1], file_path=dvc_yaml_path)

    for extra, active in extras.items():
        if active:
            try: 
                model = args.model+'_'+extra if extra != '' else args.model
                if args.action == 'get':
                    model_registry.download_model(model, args.version, remote=args.remote, remote_config=args.remote_config, submodule=args.submodule)
                elif args.action == 'add':
                    model_registry.add_model(model, args.version, push=args.push, submodule=args.submodule)
            except Exception as e:
                if args.debug:
                    print(e)
                print(f"Warning: Error while trying to {args.action} the model {extra}: {args.model} with version {args.version}")