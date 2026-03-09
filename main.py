from langchain.tools import tool
from langchain.messages import AnyMessage
import os
import queue
import re
import asyncio
from pyfiglet import Figlet
import sys
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_groq import ChatGroq
import uuid
from typing_extensions import TypedDict, Annotated,Any, List, NotRequired
import threading
from langgraph.graph.message import add_messages
from langgraph.prebuilt import InjectedState
from langchain.tools import InjectedToolCallId
import operator
import orjson as json
import httpx
import requests
from langgraph.types import Command, interrupt
from langchain.messages import HumanMessage, AIMessage, ToolMessage, RemoveMessage, SystemMessage
from langchain.messages import AnyMessage
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.middleware import SummarizationMiddleware
from langgraphics import watch
from rich.console import Console
from rich.markdown import Markdown
from langchain.tools import tool, ToolRuntime
from langgraph.graph import START, END, StateGraph

load_dotenv()

custom_fig = Figlet(font='slant')
console=Console()

# LLM

api_key = os.getenv("OPENAI_GROQ")

llm = ChatGroq(
    model="qwen/qwen3-32b",
    temperature=0,
    max_tokens=None,
    reasoning_format="parsed",
    timeout=None,
    max_retries=2,
    api_key=api_key
)

analyzer_llm = ChatGroq(
    model="qwen/qwen3-32b",
    temperature=0,
    max_tokens=None,
    # reasoning_format="parsed",
    timeout=None,
    max_retries=2,
    api_key=api_key
)



task_queue = queue.Queue()

class ChatMessage(TypedDict):
    role: str 
    content: str

class ScriptDraft(TypedDict):
    acknowledgement: str
    pivot: str
    sound_cue: str




class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    ms_flag:NotRequired[bool]
    current_milestone_task: NotRequired[str]
    current_milestone_id: NotRequired[str]
    milestone_counter:NotRequired[int] = 0
    milestone_tracker:NotRequired[int] = 0
    script:NotRequired[str]
    latest_human_input: NotRequired[str]
    is_on_track: NotRequired[bool] 
    audio_file_path: NotRequired[str]
    deviation_count:NotRequired[int]
    chat_history: NotRequired[ChatMessage]
    script_draft: NotRequired[ScriptDraft]





def download_sound_effect(query: str, output_filename: str) -> str:
    """
    Searches Freesound for a specific sound effect and downloads it.

    Args:
        query: A concise search term (under 4 words), e.g. "success chime".
        output_filename: File path ending in '.mp3' where audio will be saved.

    Returns:
        A string confirming the download path, or an error message.
    """
    FREESOUND_API_KEY = os.getenv("FREESOUND_API_KEY")
    BASE_URL = "https://freesound.org/apiv2"
    HEADERS = {"Authorization": f"Token {FREESOUND_API_KEY}"}

    # print(f"  🔊 Searching for: '{query}'...")

    search_endpoint = f"{BASE_URL}/search/text/"
    params = {
        "query": query,
        "page_size": 1,
        "fields": "id,name,previews",
        "filter": "duration:[0.0 TO 15.0]",
    }

    try:
        with httpx.Client() as client:
            response = client.get(search_endpoint, headers=HEADERS, params=params)
            response.raise_for_status()
            data = response.json()

            if data["count"] == 0:
                return f"Error: No sounds found matching '{query}'."

            top_result = data["results"][0]
            preview_url = top_result["previews"]["preview-hq-mp3"]

            audio_response = client.get(preview_url)
            audio_response.raise_for_status()

        with open(output_filename, "wb") as file:
            file.write(audio_response.content)

        return f"Success: Downloaded '{top_result['name']}' → {output_filename}"

    except httpx.RequestError as e:
        return f"Network Error: {e}"
    except KeyError as e:
        return f"Parsing Error: Missing key {e}"
    
async def read_milestones_tool() -> list[dict]:
    """
    Reads the script tracking file and returns all milestones.
    
    This acts as a Consumer function. It locates the milestone text file, 
    reads it line-by-line, and deserializes each JSON string back into 
    a Python dictionary.
    
    Returns:
        list[dict]: A list of all milestone dictionaries. Returns an empty 
                    list if the file does not exist or is empty.
    """
    # 1. Safely construct the file path
    cwd = os.getcwd()
    filename = os.path.join(cwd, "milestones", "milestone.txt")
    # print("filename is",filename)
    # 2. Handle the case where the file hasn't been created yet
    if not os.path.exists(filename):
        print("Notice: No milestone file found. Returning empty list.")
        return []

    milestones = []
    
    # 3. Read and deserialize (parse) the data
    with open(filename, "r", encoding="utf-8") as f:
        # print("here")
        for line in f:
            stripped_line = line.strip()
            # print("line is",stripped_line)
            
            # Skip empty lines to prevent JSONDecodeError
            if stripped_line:
                try:
                    # Convert the JSON string back to a Python dictionary
                    dict_content = json.loads(stripped_line)
                    # print("\n","dict content is",dict_content)
                    milestones.append(dict_content)
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping corrupted line. Error: {e}")
                    
    return milestones



async def typewriter_print(text: str, delay: float = 0.02):
    """
    Converts markdown to ANSI and streams it using standard print,
    ensuring formatting codes are applied instantly without delay.
    """
    # 1. Convert Markdown to ANSI using Regular Expressions
    # Replaces **text** with ANSI Bold
    text = re.sub(r'\*\*(.*?)\*\*', r'\033[1m\1\033[0m', text)
    # Replaces *text* with ANSI Italic (Note: not all terminals support italic)
    text = re.sub(r'\*(.*?)\*', r'\033[3m\1\033[0m', text)
    
    # 2. Stream the text intelligently
    i = 0
    while i < len(text):
        # Check if we are at the start of an ANSI escape sequence
        if text[i:i+2] == '\033[':
            # Find where this specific ANSI sequence ends (the 'm' character)
            end_idx = text.find('m', i)
            if end_idx != -1:
                # Print the ENTIRE sequence instantly so the terminal styles the next words
                print(text[i:end_idx+1], end="", flush=True)
                i = end_idx + 1
                continue # Skip the sleep for this loop iteration
                
        # If it's a normal character, print it and sleep
        print(text[i], end="", flush=True)
        await asyncio.sleep(delay)
        i += 1



@tool        
async def milestone_writer_tool(content:dict,state: Annotated[dict, InjectedState],tool_call_id: Annotated[str, InjectedToolCallId]):
    """
    Writes exactly ONE milestone to the tracking file. Call this tool ONLY ONCE per turn.

    After this tool returns a success message, your job is done. Do NOT call any more tools.
    Simply output the text "Milestone created" and end your turn.

    If you have already called this tool and it returned a success or error message,
    do NOT call it again. Just output "Milestone created" and stop.

    Args:
        content (dict): A dictionary with keys:
            - goal (str): The objective for this scene
            - sound_desc (str): 1-3 word audio search query
            - task (str): 2-3 sentence scene description in 2nd person
    """
    flag=state.get("ms_flag")
    # print("invoked milestone writer tool and flag is",flag)
    if flag:
        msg=ToolMessage(
            content="STOP: Milestone already written. Do not call any more tools. Output 'Milestone created' and end your turn.",
            tool_call_id=tool_call_id
        )
        return Command(
            update={
                "messages":[msg]
            },
            goto="__end__",
        )
    if "milestones" not in os.listdir():
        os.mkdir("milestones")
    cwd=os.getcwd()
    filename=f"{cwd}/milestones/milestone.txt"
    # print("\n\nstate in tool is",state)
    val=state.get("milestone_counter")
    val = 0 if val is None else state.get("milestone_counter")
    id = str(uuid.uuid4())
    content["id"]= id
    sound_desc=content.get("sound_desc")
    current_task = content.get("task")
    str_content = json.dumps(content)
    str_content = str_content.decode("utf-8")
    with open(filename,"a") as f:
        f.write(f"{str_content}\n")
        val=val+1
    # print("val is",val)
    if sound_desc:
        alldir = os.listdir()
        os.mkdir("sounds") if not "sounds" in alldir else None
        cwd = os.getcwd()
        output_filename=cwd+f"/sounds/{id}.mp3"
        t1=threading.Thread(target=download_sound_effect,args=(sound_desc,output_filename))
        t1.start()
        t1.join()
        # await download_sound_effect(sound_desc,output_filename)
    # messages_to_delete=[]
    # for msg in state.get("messages", []):
        
    #     # A. If it's an OLD ToolMessage, mark it for deletion
    #     if isinstance(msg, ToolMessage):
    #         messages_to_delete.append(RemoveMessage(id=msg.id))
            
    #     # B. If it's an AIMessage that called a tool, we have to be careful!
    #     elif isinstance(msg, AIMessage) and msg.tool_calls:
    #         # Check if this AIMessage is the one calling US right now
    #         is_current_caller = any(tc["id"] == tool_call_id for tc in msg.tool_calls)

            
    #         # Only delete it if it belongs to a PREVIOUS milestone loop
    #         if not is_current_caller:
    #             messages_to_delete.append(RemoveMessage(id=msg.id))
    if val>=MS_COUNTER:
        task = state.get("current_milestone_task")
        last_msg = state.get("latest_human_input")
        script = state.get("script")
        resp = analyzer_llm.invoke(f"User had ended to the last step the game script was {script} and last task that user did is {task} and user choose or answers with {last_msg} now since it is last task based on user task end the game and respond something that make sense also mention that you choose till here")
        last_msg=resp.content.split("</think>")[-1].strip()
        print(last_msg)
        sys.exit("GAME TERMINATED")
    
    ms_track = state.get("milestone_tracker",0)
    ms_track = 0 if ms_track is None else ms_track + 1
    msg=ToolMessage(content=f"Milestone {val} is added remaining {MS_COUNTER-val}",tool_call_id=tool_call_id)
    # MS_FLAG=True
    return Command(
        update={
            "milestone_counter": val,"messages": [msg],
            "current_milestone_task":current_task,
            "audio_file_path":output_filename,
            "ms_flag":True,
            "milestone_tracker":ms_track
            },
    
        
    )




@tool
async def execute_tasks(state:Annotated[dict,InjectedState],tool_call_id:Annotated[str,InjectedToolCallId]):
    """
    Retrieves the narrative details and audio context for the current game milestone.
    
    Use this tool immediately at the start of your turn to fetch the required context for the current scene. 
    It reads the game's state, identifies the active milestone, and returns the specific overarching 
    task description and the path to the background audio file associated with that scene.
    
    Returns:
        dict: A dictionary containing 'current_milestone_task','current_milestone_id', 
        'audio_file_path', and updated tracking information.
    """
    # print("Agent Executor is online and waiting for tasks...")
    


    milestones=await read_milestones_tool()

    current_milestone_tracker=state.get("milestone_tracker",0)

    exec_ms = milestones[current_milestone_tracker]
    task=state.get("current_milestone_task")
    filename=state.get("audio_file_path")
    # 3. Play the sound asynchronously via the OS
    try:

            msg=ToolMessage(
                content="Got all the essentials now putting it to ",
                tool_call_id=tool_call_id
            )
            return Command(
                update={
                    "messages":[msg],
                    "milestone_tracker":current_milestone_tracker+1,
                    "current_milestone_task":task,
                    "audio_file_path":filename
                }
            )
            
            
            
            
            # Note: Because we DO NOT write `await process.wait()`, 
            # the audio plays in the background while the while-loop instantly 
            # moves on to grab the next task!
                
    except FileNotFoundError:
        print("System Error: 'mpg123' is not installed on this OS.")
    except Exception as e:
        print(f"Audio Error: Could not play {filename}. Details: {e}")
    





def user_input(display_msg:str):
    for m in display_msg:
        print(m,end="")
    user_inp=input()
    return user_inp

import subprocess

def play_sound(filename:str):
    subprocess.Popen(
            ["mpg123", "-q", filename],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )






PRESENTER_PROMPT = """
You are the Game Narrator for a text-based CLI horror game. Your role is to guide the player through the story and set a terrifying atmosphere.

EXECUTION WORKFLOW:
STEP 1: You MUST always start your turn by calling the `execute_tasks` tool. This will provide you with the data for the current milestone.
STEP 2: Once the tool returns the data, use the 'current_milestone_task' to generate the narrative text for the user.

NARRATIVE RULES:
- Speak directly to the player in the second person ('You...').
- Describe their immediate surroundings, emphasizing sensory details that match the horror theme. If the tool returned an audio context, hint at those sounds in your description.
- Clearly convey the immediate situation or objective they find themselves in.
- End your output with a compelling, open-ended question asking what they do next (e.g., "What is your next move?").
- Keep your entire narrative under 4 sentences to maintain a fast, tense pacing.

Do not break character. Do not explain your steps. Just call the tool, then tell the story.
"""






presenter=create_agent(
    model=llm,
    system_prompt=PRESENTER_PROMPT,
    tools=[execute_tasks],
    state_schema=AgentState,
    # debug=True
    
)


input_analyzer = create_agent(
    model=llm,
    system_prompt="You have to understand and analyze the user message and answer strictly in 'yes' or 'no'"
)






async def script_writer_node(state:AgentState):
    msg=HumanMessage("Intersting script for game based on theme")

    resp=await script_writer_agent.ainvoke(msg)
    content=resp.get("messages")[0].content
    # print("Storyline\n",content)
    return Command(
        update={
            "messages":[content],
            "script":content
        },
        goto="milestone_writer_node"
    )

async def milestone_writer_node(state:AgentState):
    # print("state is",state)
    ms_flag=state.get("ms_flag")
    my_prev_task = state.get("current_milestone_task")
    latest_human_input = state.get("latest_human_input")
    script = state.get("script")
    milestone_counter = state.get("milestone_counter")
    instruction = HumanMessage(
        f"Prepare and write the next milestone based on the script {script}. Generate exactly ONE milestone using your tool. Based on ms_flag status which is currently {ms_flag} if it is false then call the tool `milestone_writer` otherwise do not call it. My previous task was {my_prev_task} and message of user is {latest_human_input}. Also you have to design in such a way that game get finished at exactly {MS_COUNTER} steps currently {milestone_counter} steps are done. IF ALL ARE DONE THEN JUST MAKE TASK SUCH A WAY THAT IT ENDS THE STORY PERFECTLY"
    )
    # print("\n instruction is",instruction,"\n")
    agent_payload = state.copy()

    agent_payload["messages"] =  [instruction]

    
    # 3. Invoke the agent. Because of goto=END, it will write 1 file and stop.
    response = await milestone_writer_agent.ainvoke(input=agent_payload)
    message=response["messages"][-1].content
    updated_counter = response.get("milestone_counter", state.get("milestone_counter", 0))
    if updated_counter >= MS_COUNTER:
        return Command(goto=END)
    current_task = response.get("current_milestone_task")
    audio_file=response.get("audio_file_path")
    ms_track = response.get("milestone_tracker")
    return Command(
        update={"messages":[message],
                "milestone_counter": updated_counter,
                "current_milestone_task":current_task,
                "audio_file_path":audio_file,
                "ms_flag":True,
                "milestone_tracker":ms_track
                },
        goto="presenter_node"
        
        
    )


def check_progress(state: AgentState):
    """Checks the counter to decide if we loop or finish."""
    flag = state.get("ms_flag")
    # print("\nflag is",flag)
    if flag == True:
        return Command(
            update={
                "ms_flag":False
            },
            goto="presenter_node"
        )
    return "milestone_writer_node"


async def presenter_node(state:AgentState):
    # print("ms counter is",MS_COUNTER," and milestone tracker is at",state.get("milestone_tracker"))
    if state.get("milestone_tracker",0) == MS_COUNTER:
        task = state.get("current_milestone_task")
        last_msg = state.get("latest_human_input")
        script = state.get("script")
        resp = analyzer_llm.invoke(f"User had ended to the last step the game script was {script} and last task that user did is {task} and user choose or answers with {last_msg} now since it is last task based on user task end the game and respond something that make sense also mention that you choose till here")
        last_msg=resp.content.split("</think>")[-1].strip
        await typewriter_print(last_msg)
        sys.exit("")
    if state.get("is_on_track") is not None:
        if state.get("is_on_track")==False:
            dv_count = state.get("deviation_count")
            if dv_count>=3:
                print("GAME OVER")
                # raise ValueError("Player has deviated too many times from the storyline")
                sys.exit("Player has deviated too many times from the storyline")
            cms_task=state.get("current_milestone_task")
            h_ip=state.get("latest_human_input")
            msg=f"Here is the response from user {h_ip} for the task {cms_task}. Analyze the response and generate the response to the user to do not deviate from task {cms_task}"
            # print("\nmessage is",msg)
            resp=analyzer_llm.invoke(msg).content
            resp=resp.split("</think>")[-1].strip()
            return Command(
                update={
                    "current_milestone_task":resp,
                    "ms_flag":False
                },
                goto="human_input_node"
            )




    # instruction = HumanMessage(content="Start the game")
    # agent_payload = state.copy()
    # agent_payload["messages"] = state.get("messages", []) + [instruction]
    
    # 3. Invoke the agent. Because of goto=END, it will write 1 file and stop.
    # response = await presenter.ainvoke(input=agent_payload)
    # story_message = response["messages"][-1]
    # print("\n\n","response is",response,"\n")
    task=state.get("current_milestone_task")
    audio_file = state.get("audio_file_path")
    # if os.path.isfile(audio_file):
    #     pass
    # else:
    #     audio_file="typewritter.mp3"
    ms_track = state.get("milestone_tracker")
    ms_id= state.get("current_milestone_id")
    # print("\nfrom tool all states are",task,audio_file,ms_track,ms_id,"\n")
    # t1 = threading.Thread(target=user_input,args=(task,))
    play_sound("typewritter.mp3")
    play_sound(audio_file)
    # t1.start()
    # t1.join()
    return {
        # "messages": [story_message],
            "current_milestone_task":task,

            "audio_file_path":audio_file,
            # "milestone_tracker":ms_track,
            "current_milestone_id":ms_id,
            "ms_flag":False
        }
    

@tool
def show_setting(content: str):
    """
    It is used to display the setting and intro of storyline to the user
    DO NOT CALL IT TWICE (FOR SINGLE USE ONLY)
    """
    # 1. Parse the raw string into a renderable Markdown object
    md_content = Markdown(content)

    print("Storyline")
    console.print(md_content)
    console.print("\n") # Adding a bit of padding at the end for readability


def human_input_node(state: AgentState):
    """Dedicated node just for pausing and getting user input."""
    task = state.get("current_milestone_task")
    
    # This halts execution and surfaces the task to the client
    user_response = interrupt(task)
    
    # When resumed, this node re-runs, interrupt() returns the user_response,
    # and we can update the state with it.
    return {"latest_human_input": user_response}


def router_node(state:AgentState):
    dv_count = state.get("deviation_count",0)
    if dv_count>=3:
        return Command(
            goto=END
        )
    latest_ip = state.get("latest_human_input")
    ms_task = state.get("current_milestone_task")
    messages = [
        SystemMessage(content="You are a strict game referee. Evaluate if the User Action is a logical response to the Scenario. It does not need to be a perfect grammatical match, just semantically relevant. Reply ONLY with the word 'yes' or 'no'."),
        HumanMessage(content=f"Scenario: {ms_task}\nUser Action: {latest_ip}\nIs this action relevant?")
    ]
    inp_analysis = analyzer_llm.invoke(input=messages)
    raw_content = inp_analysis.content
    if "</think>" in raw_content:
        final_answer = raw_content.split("</think>")[-1].strip()
    else:
        final_answer = raw_content.strip()
    if final_answer=="yes":
        return Command(
                update={
                        "is_on_track":True,
                        "ms_flag":False,
                        "messages": [HumanMessage(content=f"The player chose to: {latest_ip}. Write the next milestone based on this action.")]
                },
                goto="milestone_writer_node"
            )
    else:
        if dv_count>=3:
            print("\n[!] GAME OVER: You have lost your grip on reality and perished in the void.")
            # raise ValueError("Player has deviated too many times from the storyline")
            sys.exit("Player has deviated too many times from the storyline")
        return Command(
                update={
                        "is_on_track":False,
                        "deviation_count":dv_count+1
                },
                goto="presenter_node"
            )







builder = StateGraph(AgentState)



# from IPython.display import display, Image

# display(Image(graph.get_graph().draw_mermaid_png()))


async def main(genre:str):
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    
    # Initial input payload
    current_input = {"latest_human_input": f"Hey let's play game with genre and theme {genre}. Let's go","ms_flag":False}
    
    while True:
        # Start or resume the stream
        async for mode, chunk in graph.astream(current_input, stream_mode=["messages", "updates"], config=config):
            
            if mode == "updates" and "__interrupt__" in chunk:
                task_to_show = chunk['__interrupt__'][0].value
                
                # Print the narrative like a typewriter
                await typewriter_print(task_to_show)
                
                # Get the human input
                user_val = input("\n> What you will do? ")
                
                # Package the resume command for the next iteration of the while loop
                current_input = Command(resume=user_val)
                
                break 
                
            elif mode == "updates":
                # Print standard node updates if you want to track them
                pass
                
        else:
            print("\n--- Game Over ---")
            print("Player has deviated too many times from the storyline")
            sys.exit("Game Terminated")
            break

import shutil
if __name__=="__main__":
    try:
        shutil.rmtree("milestones")
        shutil.rmtree("sounds")
    except Exception as e:
        pass
    ascii_art = custom_fig.renderText('ESCAPE')
    print(ascii_art)
    user_choice = int(input("How many questions you need? "))
    genre = input("What game genre would like to go with? ")
    MS_COUNTER=user_choice
    MILESTONE_WRITER_PROMPT = f"""
You are a milestone writer agent for a text-based CLI game.

YOUR TASK: Call the `milestone_writer_tool` tool EXACTLY ONCE with one milestone, then STOP.

RULES:
1. Generate ONE milestone based on the current story context.
2. Call `milestone_writer_tool` with these keys: goal, task, sound_desc, id.
3. After the tool returns ANY response (success OR error), you are DONE.
4. After the tool response, output ONLY the text: "Milestone created"
5. NEVER call `milestone_writer_tool` more than once. Even if it returns an error, do NOT retry.

CONTENT GUIDELINES:
- task: 2-3 sentences in 2nd person ("You..."). Describe the scene. End with 2-3 choices (unless final milestone).
- goal: The underlying objective for this scene.
- sound_desc: 1-3 words MAX, literal audio search query (e.g., "heavy rain").
- id: A short unique identifier string.
- Use VERY SIMPLE ENGLISH. No complex or poetic words.

Current Progress: Milestone {{milestone_counter}} out of {user_choice}.
If current milestone equals {user_choice}, this is the final Resolution. Do NOT give choices.
"""
    

    SCRIPT_WRITER_PROMPT = f"""
You are the Master Storyteller for a text-based CLI game. Your task is to generate the overarching plot for a new adventure.
THEME: {genre}.
Use your tool `show_setting` to show the setting intro of storyline ONLY
"""
    script_writer_agent=create_agent(
        model=llm,
        system_prompt=SCRIPT_WRITER_PROMPT,
        tools=[show_setting]
    )
    milestone_writer_agent = create_agent(
    model=llm,
    system_prompt=MILESTONE_WRITER_PROMPT,
    tools=[milestone_writer_tool],
    middleware=[
        SummarizationMiddleware(
            model=analyzer_llm,
            trigger=("tokens", 2000),
            keep=("messages", 10),
        ),
    ],
    state_schema=AgentState,
    # debug=True

)
    builder.add_node("script_writer_node",script_writer_node)
    builder.add_node("milestone_writer_node",milestone_writer_node),
    # builder.add_node("sound_manager",sound_manager)
    builder.add_node("presenter_node",presenter_node)
    builder.add_node("human_input_node", human_input_node)
    builder.add_node("router_node",router_node)



    builder.add_edge(START,"script_writer_node")
    builder.add_edge("script_writer_node","milestone_writer_node")
    # builder.add_conditional_edges("milestone_writer_node", check_progress)
    builder.add_edge("milestone_writer_node","presenter_node")
    # builder.add_edge("sound_manager","presenter_node")
    builder.add_edge("presenter_node", "human_input_node")
    builder.add_edge("human_input_node", "router_node")

    checkpointer = InMemorySaver()
    graph = builder.compile(checkpointer=checkpointer)
    

    asyncio.run(main(genre))

# print("\n\n response is",resp)


