model = LeNet5(num_classes).to(device)

cost = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
total_step = len(train_loader)
for epoch in range(num_epochs):
  for i ,(images,labels) in enumerate(train_loader):
    images = images.to(device)
    labels = labels.to(device)

    #Forward pass
    outputs = model(images)
    loss = cost(outputs,labels)
    #Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if(i+1) % 400 == 0:
        print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1,num_epochs,i+1,total_step,loss.item()))
