import torch
from mido import MidiFile
from mido import Message, MidiFile, MidiTrack
import pandas as pd
import string
from tqdm import tqdm

DATASET_PATH = "maestro-v3.0.0"

### Define Vocabulary
note_on = [f"NOTE_ON<{i}>" for i in range(128)]
note_off = [f"NOTE_OFF<{i}>" for i in range(128)]
time_shift = [f"TIME_SHIFT<{(i + 1) * 10}>" for i in range(100)]
set_velocity = [f"SET_VELOCITY<{(i) * 4}>" for i in range(32)]
vocabulary = ['<PAD>'] + note_on + note_off + time_shift + set_velocity + ['<SOS>', '<EOS>']
vocab_size = len(vocabulary)
pad_id = 0
sos_id = vocab_size - 2
eos_id = vocab_size - 1

ttoi = {t:i for i, t in enumerate(vocabulary)}
itot = {i:t for i, t in enumerate(vocabulary)}
tok_to_id = lambda x: [ttoi[t] for t in x]
id_to_tok = lambda x: [itot[i] for i in x]

### Define Tokenizer
def ticks_to_ms(ticks, ticks_per_beat, tempo):
    return (ticks / ticks_per_beat) * tempo / 1000

def ms_to_ticks(ms, ticks_per_beat, tempo):
    return int((ms * 1000) / tempo * ticks_per_beat)

def elapsed_to_tokens(elapsed_ticks, ticks_per_beat, tempo):
    res = []
    # convert ticks to ms
    elapsed_ms = int(ticks_to_ms(elapsed_ticks, ticks_per_beat, tempo))

    # use as many of the largest TIME_SHIFT token as possible
    max_shift = 1000
    if elapsed_ms >= max_shift:
        res.extend([128 + 128 + 100] * (elapsed_ms // max_shift))
        elapsed_ms %= max_shift
    
    # deal with the remaining time
    remaining_shift_id = (elapsed_ms // 10)
    if remaining_shift_id > 0:
        res.append(128 + 128 + remaining_shift_id)
    return res

def encode(filename):
    # read midi file as object
    mid = MidiFile(filename)

    # read file metadata
    ticks_per_beat = mid.ticks_per_beat
    tempo = 0
    for msg in mid.tracks[0]:
        if msg.is_meta and msg.type == 'set_tempo':
            tempo = msg.tempo
    
    # set up counter variables
    elapsed = 0
    tokens = []

    # keep track of last velocity
    last_vel = -1

    # keep track of sustained notes and pedal state
    sustained_notes = [0 for i in range(128)] # 1 if sustained, else 0
    pedal_down = False

    # loop through main track
    for msg in mid.tracks[1]:
        msg_t = msg.type
        elapsed += msg.time
        if msg_t == 'control_change' and msg.control == 64: # sustain pedal event
            pedal_down = (msg.value >= 64)
            if not pedal_down: # pedal lifted
                # end all the sustained notes
                for i in range(len(sustained_notes)):
                    if sustained_notes[i]:
                        tokens += elapsed_to_tokens(elapsed, ticks_per_beat, tempo) # TIME_SHIFT
                        elapsed = 0
                        tokens.append(1 + 128 + i) # NOTE_OFF
                        sustained_notes[i] = 0
        elif msg_t == 'note_on' and msg.velocity > 0:
            # end currently sustained notes if they are pressed again
            if sustained_notes[msg.note]:
                tokens += elapsed_to_tokens(elapsed, ticks_per_beat, tempo) # TIME_SHIFT
                elapsed = 0
                tokens.append(1 + 128 + msg.note) # NOTE_OFF
                sustained_notes[msg.note] = 0 # remove ended note from pedal notes
            tokens += elapsed_to_tokens(elapsed, ticks_per_beat, tempo) # TIME_SHIFT
            elapsed = 0 # reset elapsed variable
            if last_vel != msg.velocity:
                vel_bin = msg.velocity // 4
                tokens.append(1 + 128 + 128 + 100 + vel_bin) # SET_VELOCITY token
                last_vel = msg.velocity
            tokens.append(1 + msg.note) # NOTE_ON
        elif msg_t == 'note_off' or (msg_t == 'note_on' and msg.velocity == 0):
            if pedal_down: # if a note ends while pedal is down, mark as being sustained
                sustained_notes[msg.note] = 1
            else:
                tokens += elapsed_to_tokens(elapsed, ticks_per_beat, tempo)
                elapsed = 0
                tokens.append(1 + 128 + msg.note) # NOTE_OFF
    return tokens

def decode(tokens, filename):
    mid = MidiFile()
    track1 = MidiTrack()
    mid.tracks.append(track1)
    vel = 0
    elapsed_ms = 0
    for tok in id_to_tok(tokens):
        if "PAD" in tok:
            print("error pad encountered")
            break
        if "SOS" in tok or "EOS" in tok:
            print("eos/sos encountered")
            break
        value = int(tok.strip(string.ascii_letters + '_<>'))
        if 'SET_VELOCITY' in tok:
            vel = value
        elif 'TIME_SHIFT' in tok:
            elapsed_ms += value
        elif 'NOTE_ON' in tok:
            track1.append(Message('note_on', note=value, velocity=vel, time=ms_to_ticks(elapsed_ms, 480, 500000)))
            elapsed_ms = 0
        elif 'NOTE_OFF' in tok:
            track1.append(Message('note_off', note=value, velocity=vel, time=ms_to_ticks(elapsed_ms, 480, 500000)))
            elapsed_ms = 0
    mid.save(filename)
    print(f"saved to {filename}")
    return mid

if __name__ == "__main__":
    ### Tokenize Data
    maestro = pd.read_csv(f'{DATASET_PATH}/maestro-v3.0.0.csv')
    midi_paths_train = maestro[maestro['split'] == 'train']['midi_filename'].tolist()
    midi_paths_valid = maestro[maestro['split'] == 'validation']['midi_filename'].tolist()
    midi_paths_test = maestro[maestro['split'] == 'test']['midi_filename'].tolist()

    train_data = []
    valid_data = []
    ctx_len = 2048

    for filename in tqdm(midi_paths_train):
        train_data.append(sos_id)
        tokens = encode(f"{DATASET_PATH}/{filename}")
        train_data.extend(tokens)
        train_data.append(eos_id)

    for filename in tqdm(midi_paths_valid):
        valid_data.append(sos_id)
        tokens = encode(f"{DATASET_PATH}/{filename}")
        valid_data.extend(tokens)
        valid_data.append(eos_id)

    train_data = torch.tensor(train_data)

    valid_data = torch.tensor(valid_data)
    valid_data = valid_data[:(valid_data.shape[0] // (ctx_len + 1)) * (ctx_len + 1)]
    valid_data = valid_data.view(-1, ctx_len + 1)

    ### Save Data
    torch.save(train_data, 'train_data.pt')
    torch.save(valid_data, 'valid_data.pt')