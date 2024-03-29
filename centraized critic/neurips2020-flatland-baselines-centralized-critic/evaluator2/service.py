#!/usr/bin/env python
from __future__ import print_function

import glob
import os
import random
import shutil
import time
import traceback
import json
import itertools

import crowdai_api
import msgpack
import msgpack_numpy as m
import pickle
import numpy as np
import redis
import timeout_decorator

import flatland
from flatland.core.env_observation_builder import DummyObservationBuilder
from flatland.envs.agent_utils import RailAgentStatus
from flatland.envs.malfunction_generators import malfunction_from_file
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import rail_from_file
from flatland.envs.schedule_generators import schedule_from_file
from flatland.evaluators import aicrowd_helpers
from flatland.evaluators import messages
from flatland.utils.rendertools import RenderTool
from flatland.envs.rail_env_utils import load_flatland_environment_from_file
from flatland.envs.persistence import RailEnvPersister

use_signals_in_timeout = True
if os.name == 'nt':
    """
    Windows doesnt support signals, hence
    timeout_decorators usually fall apart.
    Hence forcing them to not using signals 
    whenever using the timeout decorator.
    """
    use_signals_in_timeout = False

m.patch()

########################################################
# CONSTANTS
########################################################
PER_STEP_TIMEOUT = 10 * 60  # 5 minutes
RANDOM_SEED = int(os.getenv("FLATLAND_EVALUATION_RANDOM_SEED", 1001))
SUPPORTED_CLIENT_VERSIONS = \
    [
        flatland.__version__
    ]


class FlatlandRemoteEvaluationService:
    """
    A remote evaluation service which exposes the following interfaces
    of a RailEnv :
    - env_create
    - env_step
    and an additional `env_submit` to cater to score computation and on-episode-complete post processings.

    This service is designed to be used in conjunction with
    `FlatlandRemoteClient` and both the srevice and client maintain a
    local instance of the RailEnv instance, and in case of any unexpected
    divergences in the state of both the instances, the local RailEnv
    instance of the `FlatlandRemoteEvaluationService` is supposed to act
    as the single source of truth.

    Both the client and remote service communicate with each other
    via Redis as a message broker. The individual messages are packed and
    unpacked with `msgpack` (a patched version of msgpack which also supports
    numpy arrays).
    """

    def __init__(self,
                 test_env_folder="/tmp",
                 flatland_rl_service_id='FLATLAND_RL_SERVICE_ID',
                 remote_host='127.0.0.1',
                 remote_port=6379,
                 remote_db=0,
                 remote_password=None,
                 visualize=False,
                 video_generation_envs=[],
                 report=None,
                 verbose=False,
                 actionDir=None,   
                 episodeDir=None,  
                 mergeDir=None,
                 use_pickle=False,
                 shuffle=True,
                 missing_only=False,
                 ):


        self.actionDir = actionDir
        if actionDir and not os.path.exists(self.actionDir):
            os.makedirs(self.actionDir)

        self.episodeDir = episodeDir
        if episodeDir and not os.path.exists(self.episodeDir):
            os.makedirs(self.episodeDir)

        self.mergeDir = mergeDir
        if mergeDir and not os.path.exists(self.mergeDir):
            os.makedirs(self.mergeDir)

        self.use_pickle = use_pickle
        self.missing_only = missing_only

        # Test Env folder Paths
        self.test_env_folder = test_env_folder
        self.video_generation_envs = video_generation_envs
        self.env_file_paths = self.get_env_filepaths()

        if shuffle:
            random.shuffle(self.env_file_paths)
        print(self.env_file_paths)
        # Shuffle all the env_file_paths for more exciting videos
        # and for more uniform time progression

        # Logging and Reporting related vars
        self.verbose = verbose
        self.report = report

        # Communication Protocol Related vars
        self.namespace = "flatland-rl"
        self.service_id = flatland_rl_service_id
        self.command_channel = "{}::{}::commands".format(
            self.namespace,
            self.service_id
        )

        # Message Broker related vars
        self.remote_host = remote_host
        self.remote_port = remote_port
        self.remote_db = remote_db
        self.remote_password = remote_password
        self.instantiate_redis_connection_pool()

        # AIcrowd evaluation specific vars
        self.oracle_events = crowdai_api.events.CrowdAIEvents(with_oracle=True)
        self.evaluation_state = {
            "state": "PENDING",
            "progress": 0.0,
            "simulation_count": 0,
            "total_simulation_count": len(self.env_file_paths),
            "score": {
                "score": 0.0,
                "score_secondary": 0.0
            },
            "meta": {
                "normalized_reward": 0.0
            }
        }
        self.stats = {}

        # RailEnv specific variables
        self.env = False
        self.env_renderer = False
        self.reward = 0
        self.simulation_count = -1
        self.simulation_rewards = []
        self.simulation_rewards_normalized = []
        self.simulation_percentage_complete = []
        self.simulation_steps = []
        self.simulation_times = []
        self.env_step_times = []
        self.begin_simulation = False
        self.current_step = 0
        self.visualize = visualize
        self.vizualization_folder_name = "./.visualizations"
        self.record_frame_step = 0

        if self.visualize:
            if os.path.exists(self.vizualization_folder_name):
                print("[WARNING] Deleting already existing visualizations folder at : {}".format(
                    self.vizualization_folder_name
                ))
                shutil.rmtree(self.vizualization_folder_name)
            os.mkdir(self.vizualization_folder_name)
        


    def update_running_mean_stats(self, key, scalar):
        """
        Computes the running mean for certain params
        """
        mean_key = "{}_mean".format(key)
        counter_key = "{}_counter".format(key)

        try:
            self.stats[mean_key] = \
                ((self.stats[mean_key] * self.stats[counter_key]) + scalar) / (self.stats[counter_key] + 1)
            self.stats[counter_key] += 1
        except KeyError:
            self.stats[mean_key] = 0
            self.stats[counter_key] = 0

    def get_env_filepaths(self):
        """
        Gathers a list of all available rail env files to be used
        for evaluation. The folder structure expected at the `test_env_folder`
        is similar to :

            .
            ├── Test_0
            │   ├── Level_1.pkl
            │   ├── .......
            │   ├── .......
            │   └── Level_99.pkl
            └── Test_1
                ├── Level_1.pkl
                ├── .......
                ├── .......
                └── Level_99.pkl
        """
        print("Test_env_folder: ", self.test_env_folder)

        env_paths = \
            sorted(glob.glob(os.path.join(self.test_env_folder,"envs/*.pkl"))) + \
            sorted(glob.glob(os.path.join(self.test_env_folder,"envs/*.mpk")))

        print("env_paths:", len(env_paths))

        # Remove the root folder name from the individual
        # lists, so that we only have the path relative
        # to the test root folder
        env_paths = sorted([os.path.relpath(
            x, self.test_env_folder
        ) for x in env_paths])

        # if requested, only generate actions for those envs which don't already have them
        if self.mergeDir and self.missing_only:
            existing_paths = (itertools.chain.from_iterable(
                [ glob.glob(os.path.join(self.mergeDir, f"envs/*.{ext}")) 
                    for ext in ["pkl", "mpk"] ]))
            existing_paths = [os.path.relpath(sPath, self.mergeDir) for sPath in existing_paths]
            env_paths = sorted(set(env_paths) - set(existing_paths))


        return env_paths

    def instantiate_redis_connection_pool(self):
        """
        Instantiates a Redis connection pool which can be used to
        communicate with the message broker
        """
        if self.verbose or self.report:
            print("Attempting to connect to redis server at {}:{}/{}".format(
                self.remote_host,
                self.remote_port,
                self.remote_db))

        self.redis_pool = redis.ConnectionPool(
            host=self.remote_host,
            port=self.remote_port,
            db=self.remote_db,
            password=self.remote_password
        )
        self.redis_conn = redis.Redis(connection_pool=self.redis_pool)

    def get_redis_connection(self):
        """
        Obtains a new redis connection from a previously instantiated
        redis connection pool
        """
        return self.redis_conn

    def _error_template(self, payload):
        """
        Simple helper function to pass a payload as a part of a
        flatland comms error template.
        """
        _response = {}
        _response['type'] = messages.FLATLAND_RL.ERROR
        _response['payload'] = payload
        return _response

    @timeout_decorator.timeout(
        PER_STEP_TIMEOUT,
        use_signals=use_signals_in_timeout)  # timeout for each command
    def _get_next_command(self, _redis):
        """
        A low level wrapper for obtaining the next command from a
        pre-agreed command channel.
        At the momment, the communication protocol uses lpush for pushing
        in commands, and brpop for reading out commands.
        """
        command = _redis.brpop(self.command_channel)[1]
        return command

    def get_next_command(self):
        """
        A helper function to obtain the next command, which transparently
        also deals with things like unpacking of the command from the
        packed message, and consider the timeouts, etc when trying to
        fetch a new command.
        """
        try:
            _redis = self.get_redis_connection()
            command = self._get_next_command(_redis)
            if self.verbose or self.report:
                print("Command Service: ", command)
        except timeout_decorator.timeout_decorator.TimeoutError:
            raise Exception(
                "Timeout in step {} of simulation {}".format(
                    self.current_step,
                    self.simulation_count
                ))
        
        if self.use_pickle:
            command = pickle.loads(command)
        else:
            command = msgpack.unpackb(
                command,
                object_hook=m.decode,
                strict_map_key=False,   # msgpack 1.0
                encoding="utf8"        # msgpack 1.0
            )
        if self.verbose:
            print("Received Request : ", command)

        message_queue_latency = time.time() - command["timestamp"]
        self.update_running_mean_stats("message_queue_latency", message_queue_latency)
        return command

    def send_response(self, _command_response, command, suppress_logs=False):
        _redis = self.get_redis_connection()
        command_response_channel = command['response_channel']

        if self.verbose and not suppress_logs:
            print("Responding with : ", _command_response)

        if self.use_pickle:
            sResponse = pickle.dumps(_command_response)
        else:
            sResponse = msgpack.packb(
                _command_response,
                default=m.encode,
                use_bin_type=True)

        _redis.rpush(command_response_channel, sResponse)
            

    def handle_ping(self, command):
        """
        Handles PING command from the client.
        """
        service_version = flatland.__version__
        if "version" in command["payload"].keys():
            client_version = command["payload"]["version"]
        else:
            # 2.1.4 -> when the version mismatch check was added
            client_version = "2.1.4"

        _command_response = {}
        _command_response['type'] = messages.FLATLAND_RL.PONG
        _command_response['payload'] = {}
        if client_version not in SUPPORTED_CLIENT_VERSIONS:
            _command_response['type'] = messages.FLATLAND_RL.ERROR
            _command_response['payload']['message'] = \
                "Client-Server Version Mismatch => " + \
                "[ Client Version : {} ] ".format(client_version) + \
                "[ Server Version : {} ] ".format(service_version)
            self.send_response(_command_response, command)
            raise Exception(_command_response['payload']['message'])

        self.send_response(_command_response, command)

    def handle_env_create(self, command):
        """
        Handles a ENV_CREATE command from the client
        TODO: Add a high level summary of everything thats happening here.
        """
        self.simulation_count += 1
        print(f"env_create - sim count: {self.simulation_count} paths: {len(self.env_file_paths)}")
        if self.simulation_count < len(self.env_file_paths):
            """
            There are still test envs left that are yet to be evaluated 
            """
            test_env_file_path = self.env_file_paths[self.simulation_count]
            print("Evaluating : {}".format(test_env_file_path))
            test_env_file_path = os.path.join(
                self.test_env_folder,
                test_env_file_path
            )
            del self.env
            self.env = RailEnv(width=1, height=1,
                               rail_generator=rail_from_file(test_env_file_path),
                               schedule_generator=schedule_from_file(test_env_file_path),
                               malfunction_generator_and_process_data=malfunction_from_file(test_env_file_path),
                               obs_builder_object=DummyObservationBuilder(),
                               record_steps=True)

            if self.begin_simulation:
                # If begin simulation has already been initialized
                # atleast once
                self.simulation_times.append(time.time() - self.begin_simulation)
            self.begin_simulation = time.time()

            self.simulation_rewards.append(0)
            self.simulation_rewards_normalized.append(0)
            self.simulation_percentage_complete.append(0)
            self.simulation_steps.append(0)

            self.current_step = 0

            _observation, _info = self.env.reset(
                regenerate_rail=True,
                regenerate_schedule=True,
                activate_agents=False,
                random_seed=RANDOM_SEED
            )

            if self.visualize:
                if self.env_renderer:
                    del self.env_renderer
                self.env_renderer = RenderTool(self.env, gl="PGL", )

            _command_response = {}
            _command_response['type'] = messages.FLATLAND_RL.ENV_CREATE_RESPONSE
            _command_response['payload'] = {}
            _command_response['payload']['observation'] = _observation
            _command_response['payload']['env_file_path'] = self.env_file_paths[self.simulation_count]
            _command_response['payload']['info'] = _info
            _command_response['payload']['random_seed'] = RANDOM_SEED
        else:
            """
            All test env evaluations are complete
            """
            _command_response = {}
            _command_response['type'] = messages.FLATLAND_RL.ENV_CREATE_RESPONSE
            _command_response['payload'] = {}
            _command_response['payload']['observation'] = False
            _command_response['payload']['env_file_path'] = False
            _command_response['payload']['info'] = False
            _command_response['payload']['random_seed'] = False

        self.send_response(_command_response, command)
        #####################################################################
        # Update evaluation state
        #####################################################################
        progress = np.clip(
            self.simulation_count * 1.0 / len(self.env_file_paths),
            0, 1)
        mean_reward = round(np.mean(self.simulation_rewards), 2)
        mean_normalized_reward = round(np.mean(self.simulation_rewards_normalized), 2)
        mean_percentage_complete = round(np.mean(self.simulation_percentage_complete), 3)
        self.evaluation_state["state"] = "IN_PROGRESS"
        self.evaluation_state["progress"] = progress
        self.evaluation_state["simulation_count"] = self.simulation_count
        self.evaluation_state["score"]["score"] = mean_percentage_complete
        self.evaluation_state["score"]["score_secondary"] = mean_reward
        self.evaluation_state["meta"]["normalized_reward"] = mean_normalized_reward
        self.handle_aicrowd_info_event(self.evaluation_state)
        self.lActions = []


    def handle_env_step(self, command):
        """
        Handles a ENV_STEP command from the client
        TODO: Add a high level summary of everything thats happening here.
        """
        _payload = command['payload']

        if not self.env:
            raise Exception(
                "env_client.step called before env_client.env_create() call")
        if self.env.dones['__all__']:
            raise Exception(
                "Client attempted to perform an action on an Env which \
                has done['__all__']==True")

        action = _payload['action']
        time_start = time.time()
        _observation, all_rewards, done, info = self.env.step(action)
        time_diff = time.time() - time_start
        self.update_running_mean_stats("internal_env_step_time", time_diff)

        cumulative_reward = sum(all_rewards.values())
        self.simulation_rewards[-1] += cumulative_reward
        self.simulation_steps[-1] += 1
        """
        The normalized rewards normalize the reward for an 
        episode by dividing the whole reward by max-time-steps 
        allowed in that episode, and the number of agents present in 
        that episode
        """
        self.simulation_rewards_normalized[-1] += \
            cumulative_reward / (
                self.env._max_episode_steps +
                self.env.get_num_agents()
            )

        # record the actions before checking for done
        if self.actionDir is not None:
            self.lActions.append(action)

        if done["__all__"]:
            # Compute percentage complete
            complete = 0
            for i_agent in range(self.env.get_num_agents()):
                agent = self.env.agents[i_agent]
                if agent.status in [RailAgentStatus.DONE_REMOVED]:
                    complete += 1
            percentage_complete = complete * 1.0 / self.env.get_num_agents()
            self.simulation_percentage_complete[-1] = percentage_complete

            if self.actionDir is not None:
                self.save_actions()
            
            if self.episodeDir is not None:
                self.save_episode()

            if self.mergeDir is not None:
                self.save_merged_env()


        # Record Frame
        if self.visualize:
            self.env_renderer.render_env(
                show=False,
                show_observations=False,
                show_predictions=False
            )
            """
            Only save the frames for environments which are separately provided 
            in video_generation_indices param
            """
            current_env_path = self.env_file_paths[self.simulation_count]
            if current_env_path in self.video_generation_envs:
                self.env_renderer.gl.save_image(
                    os.path.join(
                        self.vizualization_folder_name,
                        "flatland_frame_{:04d}.png".format(self.record_frame_step)
                    ))
                self.record_frame_step += 1



    def save_actions(self):
        sfEnv = self.env_file_paths[self.simulation_count]
        
        sfActions = self.actionDir + "/" + sfEnv.replace(".pkl", ".json")

        print("env path: ", sfEnv, " sfActions:", sfActions)

        if not os.path.exists(os.path.dirname(sfActions)):
            os.makedirs(os.path.dirname(sfActions))
        
        with open(sfActions, "w") as fOut:
            json.dump(self.lActions, fOut)

        self.lActions = []
    
    def save_episode(self):
        sfEnv = self.env_file_paths[self.simulation_count]
        sfEpisode = self.episodeDir + "/" + sfEnv
        print("env path: ", sfEnv, " sfEpisode:", sfEpisode)
        RailEnvPersister.save_episode(self.env, sfEpisode)
        #self.env.save_episode(sfEpisode)
    
    def save_merged_env(self):
        sfEnv = self.env_file_paths[self.simulation_count]
        sfMergeEnv = self.mergeDir + "/" + sfEnv

        if not os.path.exists(os.path.dirname(sfMergeEnv)):
            os.makedirs(os.path.dirname(sfMergeEnv))

        print("Input env path: ", sfEnv, " Merge File:", sfMergeEnv)
        RailEnvPersister.save_episode(self.env, sfMergeEnv)
        #self.env.save_episode(sfMergeEnv)


    def handle_env_submit(self, command):
        """
        Handles a ENV_SUBMIT command from the client
        TODO: Add a high level summary of everything thats happening here.
        """
        _payload = command['payload']

        ######################################################################
        # Print Local Stats
        ######################################################################
        print("=" * 100)
        print("=" * 100)
        print("## Server Performance Stats")
        print("=" * 100)
        for _key in self.stats:
            if _key.endswith("_mean"):
                print("\t - {}\t:{}".format(_key, self.stats[_key]))
        print("=" * 100)

        # Register simulation time of the last episode
        self.simulation_times.append(time.time() - self.begin_simulation)

        if len(self.simulation_rewards) != len(self.env_file_paths):
            raise Exception(
                """env.submit called before the agent had the chance 
                to operate on all the test environments.
                """
            )

        mean_reward = round(np.mean(self.simulation_rewards), 2)
        mean_normalized_reward = round(np.mean(self.simulation_rewards_normalized), 2)
        mean_percentage_complete = round(np.mean(self.simulation_percentage_complete), 3)

        if self.visualize and len(os.listdir(self.vizualization_folder_name)) > 0:
            # Generate the video
            #
            # Note, if you had depdency issues due to ffmpeg, you can
            # install it by :
            #
            # conda install -c conda-forge x264 ffmpeg

            print("Generating Video from thumbnails...")
            video_output_path, video_thumb_output_path = \
                aicrowd_helpers.generate_movie_from_frames(
                    self.vizualization_folder_name
                )
            print("Videos : ", video_output_path, video_thumb_output_path)
            # Upload to S3 if configuration is available
            if aicrowd_helpers.is_grading() and aicrowd_helpers.is_aws_configured() and self.visualize:
                video_s3_key = aicrowd_helpers.upload_to_s3(
                    video_output_path
                )
                video_thumb_s3_key = aicrowd_helpers.upload_to_s3(
                    video_thumb_output_path
                )
                static_thumbnail_s3_key = aicrowd_helpers.upload_random_frame_to_s3(
                    self.vizualization_folder_name
                )
                self.evaluation_state["score"]["media_content_type"] = "video/mp4"
                self.evaluation_state["score"]["media_large"] = video_s3_key
                self.evaluation_state["score"]["media_thumbnail"] = video_thumb_s3_key

                self.evaluation_state["meta"]["static_media_frame"] = static_thumbnail_s3_key
            else:
                print("[WARNING] Ignoring uploading of video to S3")

        _command_response = {}
        _command_response['type'] = messages.FLATLAND_RL.ENV_SUBMIT_RESPONSE
        _payload = {}
        _payload['mean_reward'] = mean_reward
        _payload['mean_normalized_reward'] = mean_normalized_reward
        _payload['mean_percentage_complete'] = mean_percentage_complete
        _command_response['payload'] = _payload
        self.send_response(_command_response, command)

        #####################################################################
        # Update evaluation state
        #####################################################################
        self.evaluation_state["state"] = "FINISHED"
        self.evaluation_state["progress"] = 1.0
        self.evaluation_state["simulation_count"] = self.simulation_count
        self.evaluation_state["score"]["score"] = mean_percentage_complete
        self.evaluation_state["score"]["score_secondary"] = mean_reward
        self.evaluation_state["meta"]["normalized_reward"] = mean_normalized_reward
        self.handle_aicrowd_success_event(self.evaluation_state)
        print("#" * 100)
        print("EVALUATION COMPLETE !!")
        print("#" * 100)
        print("# Mean Reward : {}".format(mean_reward))
        print("# Mean Normalized Reward : {}".format(mean_normalized_reward))
        print("# Mean Percentage Complete : {}".format(mean_percentage_complete))
        print("#" * 100)
        print("#" * 100)

    def report_error(self, error_message, command_response_channel):
        """
        A helper function used to report error back to the client
        """
        _redis = self.get_redis_connection()
        _command_response = {}
        _command_response['type'] = messages.FLATLAND_RL.ERROR
        _command_response['payload'] = error_message
        _redis.rpush(
            command_response_channel,
            msgpack.packb(
                _command_response,
                default=m.encode,
                use_bin_type=True)
        )
        self.evaluation_state["state"] = "ERROR"
        self.evaluation_state["error"] = error_message
        self.handle_aicrowd_error_event(self.evaluation_state)

    def handle_aicrowd_info_event(self, payload):
        self.oracle_events.register_event(
            event_type=self.oracle_events.CROWDAI_EVENT_INFO,
            payload=payload
        )

    def handle_aicrowd_success_event(self, payload):
        self.oracle_events.register_event(
            event_type=self.oracle_events.CROWDAI_EVENT_SUCCESS,
            payload=payload
        )

    def handle_aicrowd_error_event(self, payload):
        self.oracle_events.register_event(
            event_type=self.oracle_events.CROWDAI_EVENT_ERROR,
            payload=payload
        )

    def run(self):
        """
        Main runner function which waits for commands from the client
        and acts accordingly.
        """
        print("Listening at : ", self.command_channel)
        MESSAGE_QUEUE_LATENCY = []
        while True:
            command = self.get_next_command()
            if "timestamp" in command.keys():
                latency = time.time() - command["timestamp"]
                MESSAGE_QUEUE_LATENCY.append(latency)

            if self.verbose:
                print("Self.Reward : ", self.reward)
                print("Current Simulation : ", self.simulation_count)
                if self.env_file_paths and \
                    self.simulation_count < len(self.env_file_paths):
                    print("Current Env Path : ",
                          self.env_file_paths[self.simulation_count])

            try:
                if command['type'] == messages.FLATLAND_RL.PING:
                    """
                        INITIAL HANDSHAKE : Respond with PONG
                    """
                    self.handle_ping(command)

                elif command['type'] == messages.FLATLAND_RL.ENV_CREATE:
                    """
                        ENV_CREATE

                        Respond with an internal _env object
                    """
                    self.handle_env_create(command)
                elif command['type'] == messages.FLATLAND_RL.ENV_STEP:
                    """
                        ENV_STEP

                        Request : Action dict
                        Respond with updated [observation,reward,done,info] after step
                    """
                    self.handle_env_step(command)
                elif command['type'] == messages.FLATLAND_RL.ENV_SUBMIT:
                    """
                        ENV_SUBMIT

                        Submit the final cumulative reward
                    """

                    print("Overall Message Queue Latency : ", np.array(MESSAGE_QUEUE_LATENCY).mean())
                    self.handle_env_submit(command)
                else:
                    _error = self._error_template(
                        "UNKNOWN_REQUEST:{}".format(
                            str(command)))
                    if self.verbose:
                        print("Responding with : ", _error)
                    self.report_error(
                        _error,
                        command['response_channel'])
                    return _error
            except Exception as e:
                print("Error : ", str(e))
                print(traceback.format_exc())
                self.report_error(
                    self._error_template(str(e)),
                    command['response_channel'])
                return self._error_template(str(e))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Submit the result to AIcrowd')
    parser.add_argument('--service_id',
                        dest='service_id',
                        default='FLATLAND_RL_SERVICE_ID',
                        required=False)

    parser.add_argument('--test_folder',
                        dest='test_folder',
                        default="../../../submission-scoring/Envs-Small",
                        help="Folder containing the files for the test envs",
                        required=False)

    parser.add_argument('--actionDir',
                        dest='actionDir',
                        default=None,
                        help="deprecated - use mergeDir.  Folder containing the files for the test envs",
                        required=False)
    
    parser.add_argument('--episodeDir',
                        dest='episodeDir',
                        default=None,
                        help="deprecated - use mergeDir.   Folder containing the files for the test envs",
                        required=False)
    
    parser.add_argument('--mergeDir',
                        dest='mergeDir',
                        default=None,
                        help="Folder to store merged envs, actions, episodes.",
                        required=False)

    parser.add_argument('--pickle',
                        default=False,
                        action="store_true",
                        help="use pickle instead of msgpack",
                        required=False)

    parser.add_argument('--noShuffle',
                        default=False,
                        action="store_true",
                        help="don't shuffle the envs.  Default is to shuffle.",
                        required=False)

    parser.add_argument('--missingOnly',
                        default=False,
                        action="store_true",
                        help="only request the envs/actions which are missing",
                        required=False)


    parser.add_argument('--verbose',
                        default=False,
                        action="store_true",
                        help="verbose debug messages",
                        required=False)

    args = parser.parse_args()

    test_folder = args.test_folder

    grader = FlatlandRemoteEvaluationService(
        test_env_folder=test_folder,
        flatland_rl_service_id=args.service_id,
        verbose=args.verbose,
        visualize=True,
        video_generation_envs=["Test_0/Level_1.pkl"],
        actionDir=args.actionDir,
        episodeDir=args.episodeDir,
        mergeDir=args.mergeDir,
        use_pickle=args.pickle,
        shuffle=not args.noShuffle,
        missing_only=args.missingOnly,
    )

    result = grader.run()
    if result['type'] == messages.FLATLAND_RL.ENV_SUBMIT_RESPONSE:
        cumulative_results = result['payload']
    elif result['type'] == messages.FLATLAND_RL.ERROR:
        error = result['payload']
        raise Exception("Evaluation Failed : {}".format(str(error)))
    else:
        # Evaluation failed
        print("Evaluation Failed : ", result['payload'])
