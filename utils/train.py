from model.transformer_model import FinalModel
import torch.nn.functional as F
import torch

import json

with open("data/data.json", encoding="utf-8-sig") as jsonFile:
    data = json.load(jsonFile)


EPOCHS = 2000
BATCH = 4

model = FinalModel().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)

for epoch in range(EPOCHS):

    xb, yb = [], []
    for inpux in range(BATCH):
        # artistname = random.choice([*data.keys()])
        artistname = random.choice(['edsheeran', 'maroon5', 'coldplay', 'charlieputh'])
        song = random.choice(data[artistname])
        title = song['title']
        lyrics = song['lyrics']

        # token = ['<startartist>', artistname, '<endartist>'] + ['<starttitle>'] + re.split("(\W)", title) + ['<endtitle>'] + re.split("(\W)", lyrics)  + ['<end>']
        token =  ['<start>'] + re.split("(\W)", lyrics)  + ['<end>']
        
        tk_idx = []
        for tk in token:
            idx = tokens['token'].index(tk)
            tk_idx.append(idx)

        # print(['<startartist>', artistname, '<endartist>'] + ['<starttitle>'])
        if len(tk_idx) <  ntokens:

            xb.append(tk_idx + [11156] * (ntokens-len(tk_idx)))
    
            yb.append(tk_idx[1:] + [11156] * (ntokens-len(tk_idx)+1))
        else:
           
            print(title, artistname, len(tk_idx))

    xb = torch.tensor(xb).to(device)
    yb = torch.tensor(yb).to(device)

    # print(xb.shape, yb.shape)
    B, T = xb.shape
    C = vocab_size
    

    logits = model(xb)
    logits = logits.view(B*T, C)
    yb = yb.view(B*T)
    loss = F.cross_entropy(logits, yb)
    
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    print(f'Epoch: {epoch}/{EPOCHS} loss: {loss.item()}')
    f = open(f"loss.txt", "a+")
    f.write(str(loss.item())+"\n")
    f.close()
    torch.cuda.empty_cache()

    if epoch % 100 == 0:
        print()
        print("Saving !!!!!!!!!!!!!!!!!!!!!!!!!")
        
        torch.save(model.state_dict(), f'50model.pt')
        print("Completed !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print()

        
        
            
        # print(song['lyrics'], len(re.split("(\W)", song['lyrics'])))